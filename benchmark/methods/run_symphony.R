suppressPackageStartupMessages({
  library(jsonlite)
})

if (dir.exists(".r_libs")) {
  .libPaths(c(normalizePath(".r_libs"), .libPaths()))
}

suppressPackageStartupMessages({
  library(symphony)
})

source("benchmark/methods/io_h5ad.R")

.parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2 || args[[1]] != "--config") {
    stop("Usage: Rscript benchmark/methods/run_symphony.R --config <config.json>")
  }
  args[[2]]
}

.shared_subset <- function(ref_sce, query_sce) {
  shared <- intersect(rownames(ref_sce), rownames(query_sce))
  if (length(shared) < 2) {
    stop(sprintf("Too few shared genes for Symphony: %d", length(shared)))
  }
  list(ref = ref_sce[shared, ], query = query_sce[shared, ], n_shared = length(shared))
}

main <- function() {
  cfg <- fromJSON(.parse_args(), simplifyVector = TRUE)
  t0 <- proc.time()[["elapsed"]]

  ref_sce <- LoadH5adToSCE(cfg$reference_h5ad, layer = cfg$reference_layer)
  query_sce <- LoadH5adToSCE(cfg$query_h5ad, layer = cfg$query_layer)
  subsetted <- .shared_subset(ref_sce, query_sce)
  ref_sce <- subsetted$ref
  query_sce <- subsetted$query

  label_col <- cfg$target_label_column
  batch_key <- cfg$batch_key
  ref_meta <- as.data.frame(SummarizedExperiment::colData(ref_sce))
  query_meta <- as.data.frame(SummarizedExperiment::colData(query_sce))
  ref_labels <- as.character(ref_meta[[label_col]])
  keep <- !is.na(ref_labels) & nzchar(ref_labels)
  if (!any(keep)) {
    stop(sprintf("No valid labels found in reference column: %s", label_col))
  }
  ref_meta <- ref_meta[keep, , drop = FALSE]
  ref_labels <- ref_labels[keep]
  ref_expr <- as.matrix(SummarizedExperiment::assay(ref_sce[, keep], "counts"))
  query_expr <- as.matrix(SummarizedExperiment::assay(query_sce, "counts"))

  vars <- NULL
  if (!is.null(batch_key) && nzchar(batch_key) && batch_key %in% colnames(ref_meta) && batch_key %in% colnames(query_meta)) {
    vars <- batch_key
  }

  chosen_vargenes_method <- as.character(cfg$vargenes_method)
  ref_obj <- tryCatch(
    buildReference(
      exp_ref = ref_expr,
      metadata_ref = ref_meta,
      vars = vars,
      K = as.integer(cfg$K),
      verbose = FALSE,
      do_umap = FALSE,
      do_normalize = isTRUE(cfg$do_normalize),
      vargenes_method = chosen_vargenes_method,
      topn = as.integer(cfg$topn),
      d = as.integer(cfg$d),
      seed = as.integer(cfg$seed)
    ),
    error = function(e) {
      if (identical(chosen_vargenes_method, "vst") && grepl("span is too small", conditionMessage(e), fixed = TRUE)) {
        chosen_vargenes_method <<- "mvp"
        buildReference(
          exp_ref = ref_expr,
          metadata_ref = ref_meta,
          vars = vars,
          K = as.integer(cfg$K),
          verbose = FALSE,
          do_umap = FALSE,
          do_normalize = isTRUE(cfg$do_normalize),
          vargenes_method = "mvp",
          topn = as.integer(cfg$topn),
          d = as.integer(cfg$d),
          seed = as.integer(cfg$seed)
        )
      } else {
        stop(e)
      }
    }
  )
  query_obj <- mapQuery(
    exp_query = query_expr,
    metadata_query = query_meta,
    ref_obj = ref_obj,
    vars = vars,
    verbose = FALSE,
    do_normalize = isTRUE(cfg$do_normalize),
    do_umap = FALSE,
    sigma = as.numeric(cfg$sigma)
  )
  save_as <- "cell_type_pred"
  query_obj <- knnPredict(
    query_obj = query_obj,
    ref_obj = ref_obj,
    train_labels = ref_labels,
    k = as.integer(cfg$knn_k),
    save_as = save_as,
    confidence = TRUE,
    seed = as.integer(cfg$seed)
  )

  prob_col <- paste0(save_as, "_prob")
  pred <- as.character(query_obj$meta_data[[save_as]])
  conf <- as.numeric(query_obj$meta_data[[prob_col]])
  predictions <- data.frame(
    cell_id = rownames(query_obj$meta_data),
    predicted_label = pred,
    conf = conf,
    margin = conf,
    is_unknown = FALSE,
    stringsAsFactors = FALSE
  )
  write.csv(predictions, cfg$output_predictions_csv, row.names = FALSE)

  metadata <- list(
    n_reference_cells = ncol(ref_expr),
    n_query_cells = ncol(query_expr),
    n_shared_genes = subsetted$n_shared,
    target_label_column = label_col,
    batch_key = if (is.null(vars)) NA else vars,
    K = as.integer(cfg$K),
    d = as.integer(cfg$d),
    knn_k = as.integer(cfg$knn_k),
    sigma = as.numeric(cfg$sigma),
    vargenes_method = chosen_vargenes_method,
    runtime_seconds = proc.time()[["elapsed"]] - t0
  )
  write_json(metadata, cfg$output_metadata_json, auto_unbox = TRUE, pretty = TRUE)
}

main()
