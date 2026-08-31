#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.samplesheet = null
params.results_dir = "${projectDir}/results"

process INGEST {
    tag "${meta.data_id}"
    publishDir "${params.results_dir}/${meta.data_id}", mode: 'copy'

    memory { meta.memory_spec }
    queue { meta.partition_spec }

    input:
    tuple val(meta), path(h5ad_path)
    
    
    output:
    tuple val(meta.data_id), path("${meta.data_id}_ingested.h5ad"), emit: ingested
    path "figures/**", emit: figures, optional: true

    script:
    def cxg_arg = (meta.cxg_flag == 'True') ? '--cxg' : ''
    """
    source myconda; conda activate nsforestv4.1

    python ${projectDir}/src/ingest.py \\
        --data_id "${meta.data_id}" \\
        --data_path "${h5ad_path}" \\
        --results_dir . \\
        --cluster_header "${meta.cluster_header}" \\
        --tmpdir . \\
        --var_col "${meta.var_col}" \\
        --cluster_labels "${meta.cluster_labels}" \\
        ${cxg_arg}
    """
}

workflow {
    if (!params.samplesheet) {
        error "Provide --samplesheet <path to tsv>"
    }

    Channel
        .fromPath(params.samplesheet)
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
        .set { ingest_inputs }

    INGEST(ingest_inputs)
}