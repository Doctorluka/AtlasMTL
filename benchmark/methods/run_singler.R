suppressPackageStartupMessages({
  library(jsonlite)
  library(SingleR)
  library(scuttle)
})

source("benchmark/methods/io_h5ad.R")

.parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2 || args[[1]] != "--config") {
    stop("Usage: Rscript benchmark/methods/run_singler.R --config <config.json>")
  }
  args[[2]]
}

.ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required R package not installed: %s", pkg))
  }
}

.shared_subset <- function(ref_sce, query_sce) {
  shared <- intersect(rownames(ref_sce), rownames(query_sce))
  if (length(shared) < 2) {
    stop(sprintf("Too few shared genes for SingleR: %d", length(shared)))
  }
  list(ref = ref_sce[shared, ], query = query_sce[shared, ], n_shared = length(shared))
}

.score_to_conf <- function(x) {
  x[is.na(x)] <- 0
  1 - exp(-pmax(x, 0))
}

main <- function() {
  .ensure_package("SingleR")
  .ensure_package("scuttle")
  .ensure_package("jsonlite")
  .ensure_package("SingleCellExperiment")
  config_path <- .parse_args()
  cfg <- fromJSON(config_path, simplifyVector = TRUE)

  t0 <- proc.time()[["elapsed"]]
  ref_sce <- LoadH5adToSCE(cfg$reference_h5ad, layer = cfg$reference_layer)
  query_sce <- LoadH5adToSCE(cfg$query_h5ad, layer = cfg$query_layer)
  subsetted <- .shared_subset(ref_sce, query_sce)
  ref_sce <- subsetted$ref
  query_sce <- subsetted$query

  label_col <- cfg$target_label_column
  ref_labels <- as.character(SummarizedExperiment::colData(ref_sce)[[label_col]])
  keep <- !is.na(ref_labels) & nzchar(ref_labels)
  if (!any(keep)) {
    stop(sprintf("No valid labels found in reference column: %s", label_col))
  }
  ref_sce <- ref_sce[, keep]
  ref_labels <- ref_labels[keep]

  if (isTRUE(cfg$normalize_log1p)) {
    ref_sce <- scuttle::logNormCounts(ref_sce)
    query_sce <- scuttle::logNormCounts(query_sce)
  } else {
    SummarizedExperiment::assay(ref_sce, "logcounts") <- NormalizeLog1pMatrix(SummarizedExperiment::assay(ref_sce, "counts"))
    SummarizedExperiment::assay(query_sce, "logcounts") <- NormalizeLog1pMatrix(SummarizedExperiment::assay(query_sce, "counts"))
  }

  result <- SingleR::SingleR(
    test = query_sce,
    ref = ref_sce,
    labels = ref_labels,
    fine.tune = isTRUE(cfg$fine_tune),
    prune = isTRUE(cfg$prune),
    quantile = as.numeric(cfg$quantile),
    de.method = as.character(cfg$de_method),
    assay.type.test = "logcounts",
    assay.type.ref = "logcounts"
  )

  raw_label <- as.character(result$labels)
  pruned_label <- as.character(result$pruned.labels)
  delta_next <- as.numeric(result$delta.next)
  use_pruned <- isTRUE(cfg$use_pruned_labels)
  predicted_label <- if (use_pruned) pruned_label else raw_label
  is_unknown <- is.na(predicted_label) | !nzchar(predicted_label)
  predicted_label[is_unknown] <- "Unknown"

  predictions <- data.frame(
    cell_id = rownames(result),
    raw_label = raw_label,
    pruned_label = pruned_label,
    predicted_label = predicted_label,
    delta_next = delta_next,
    conf = .score_to_conf(delta_next),
    margin = ifelse(is.na(delta_next), 0, delta_next),
    is_unknown = is_unknown,
    stringsAsFactors = FALSE
  )
  write.csv(predictions, cfg$output_predictions_csv, row.names = FALSE)

  runtime_seconds <- proc.time()[["elapsed"]] - t0
  metadata <- list(
    n_reference_cells = ncol(ref_sce),
    n_query_cells = ncol(query_sce),
    n_shared_genes = subsetted$n_shared,
    target_label_column = label_col,
    use_pruned_labels = use_pruned,
    unknown_count = sum(is_unknown),
    unknown_rate = mean(is_unknown),
    runtime_seconds = runtime_seconds
  )
  write_json(metadata, cfg$output_metadata_json, auto_unbox = TRUE, pretty = TRUE)
}

main()
