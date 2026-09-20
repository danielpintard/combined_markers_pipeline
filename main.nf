#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.samplesheet = null
params.results_dir = "${projectDir}/results"

process INGEST {
    tag "${meta.data_id}"
    publishDir "${params.results_dir}/${meta.data_id}", mode: 'copy', pattern: "ingested_h5ads/*.h5ad"
    publishDir "${params.results_dir}/${meta.data_id}", mode: 'copy', pattern: "figures/**"

    input:
    tuple val(meta), path(h5ad_path) 

    memory { meta.memory_spec }
    queue { meta.partition_spec }

    output:
    tuple val(meta), path("ingested_h5ads/${meta.data_id}_ingested.h5ad"), emit: ingested
    path "figures/**", emit: figures, optional: true

    script:
    def cxg_arg = (meta.cxg_flag == 'True') ? '--cxg' : ''
    """
    source myconda; conda activate nsForest_env4.2

    python ${projectDir}/src/ingest.py \\
        --data_id "${meta.data_id}" \\
        --data_path "${h5ad_path}" \\
        --results_dir . \\
        --cluster_header "${meta.cluster_header}" \\
        --var_col "${meta.var_col}" \\
        --cluster_labels "${meta.cluster_labels}" \\
        ${cxg_arg}
    """
}

process GET_AND_EVAL_MARKERS {

    tag "${meta.data_id}"
    publishDir "${params.results_dir}/${meta.data_id}", mode: 'copy', pattern: "tables/**"
    // publishDir "${params.results_dir}/${meta.data_id}", mode: 'copy', pattern: "figures/**"
    memory { meta.memory_spec }
    queue { meta.partition_spec }

    input:
    tuple val(meta), path(ingested_h5ad_path)

    output:
    tuple val(meta), path("tables/**"), emit: tables_path
    path "tables/master_results_sheet.csv", emit: master_sheet_path

    script:
    """
    source myconda; conda activate nsForest_env4.2

    python ${projectDir}/src/get_markers_and_eval.py \\
        --data_id "${meta.data_id}" \\
        --path_to_ingested_h5ad "${ingested_h5ad_path}" \\
        --cluster_header "${meta.cluster_header}" \\
        --binary_thresholding "${meta.binary_thresholding}" \\
        --results_dir . \\
        --cluster_labels "${meta.cluster_labels}" \\
        --n_cores "${task.cpus}"
    """
}

process REPORTING {
    tag "${meta.data_id}"
    publishDir "${params.results_dir}/${meta.data_id}", mode: 'copy', pattern: "figures/**"
    publishDir "${params.results_dir}/${meta.data_id}", mode: 'copy', pattern: "tables/**"

    input:
    tuple val(meta), path(ingested_h5ad_path), path(master_csv)

    output:
    path "figures/**", emit: fig_path, optional: true
    path "tables/**", emit: tables_path, optional: true

    script:
    """
    source myconda; conda activate nsForest_env4.2
    python ${projectDir}/src/plots_and_reports.py \\
        --data_id "${meta.data_id}" \\
        --path_to_ingested_h5ad "${ingested_h5ad_path}" \\
        --cluster_header "${meta.cluster_header}" \\
        --results_dir . \\
        --cluster_labels "${meta.cluster_labels}" \\
        --master_results_filepath "${master_csv}"
    """
}

workflow {
    if (!params.samplesheet) {
        error "Provide --samplesheet <path to tsv>"
    }

    ingest_inputs = Channel.fromPath(params.samplesheet)
        .ifEmpty { exit 1, "Cannot find sample sheet TSV: ${params.samplesheet}"}
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            def meta = [
                data_id : row.data_id,
                cxg_flag : row.cxg_flag,
                var_col: row.var_col_arg,
                cluster_header: row.cluster_header_arg,
                cluster_labels: row.endo_labels,
                binary_thresholding: row.binary_thresholding,
                memory_spec : row.memory_spec?.replaceAll(/(?i)\s*g$/, ' GB'),
                partition_spec: row.partition_spec
            ]
            tuple(meta, file(row.h5ad_path))
        }

    INGEST(ingest_inputs)

    GET_AND_EVAL_MARKERS(INGEST.out.ingested)

    ingested_keyed = INGEST.out.ingested.map { meta, h5ad -> tuple(meta.data_id, meta, h5ad) }

    master_keyed = GET_AND_EVAL_MARKERS.out.master_sheet_path.map { meta, master_csv -> tuple(meta.data_id, master_csv) }

    reporting_inputs = ingested_keyed
        .join(master_keyed)
        .map { data_id, meta, h5ad, master_csv ->
            tuple(meta, h5ad, master_csv)
        }
    
    REPORTING(reporting_inputs)
}