#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.samplesheet = null
params.results_dir = "${projectDir}/results"

process INGEST {
    tag "${data_id}"
    publishDir "${params.results_dir}/${data_id}", mode: 'copy'

    memory "${memory_spec}"

    input:
    tuple val(data_id),
    path(h5ad_path),
    val(cxg_flag),
    val(var_col),
    val(cluster_header),
    val(cluster_labels),
    val(binary_thresholding),
    val(memory_spec)
    
    output:
    tuple val(data_id), path("${data_id}_ingested.h5ad"), emit: ingested
    path "figures/**", emit: figures, optional: true

    script:
    def cxg_arg = (cxg_flag == 'True') ? '--cxg' : ''
    """
    ingest.py \\
        --data_id "${data_id}" \\
        --data_path "${h5ad_path}" \\
        --results_dir . \\
        --cluster_header "${cluster_header}" \\
        --tmpdir . \\
        --var_col "${var_col}" \\
        --cluster_labels ${cluster_labels} \\
        ${cxg_arg}
    """
}

// ---- workflow: read the sheet, fan out over rows ----
workflow {
    if (!params.samplesheet) {
        error "Provide --samplesheet <path to tsv>"
    }

    Channel
        .fromPath(params.samplesheet)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            tuple(
                row.data_id,
                file(row.h5ad_path),
                row.cxg_flag,
                row.var_col_arg,
                row.cluster_header_arg,
                row.endo_labels.split(',').join(' '),
                row.binary_thresholding,
                row.memory_spec,
                row.partition_spec
                )
        }
        .set { ingest_inputs }

    INGEST(ingest_inputs)
}