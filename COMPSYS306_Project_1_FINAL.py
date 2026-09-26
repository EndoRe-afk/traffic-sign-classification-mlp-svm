

from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import math
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# experiment settings

SEED = 306  # fixed seed so random splits and model initialisation are reproducible
IMAGE_SIZE = (32, 32)  # common spatial resolution used for both classifiers
N_CLASSES = 43

TEST_FRACTION = 0.20  # fraction of sequence groups reserved for the untouched test split
VALIDATION_FRACTION_OF_DEVELOPMENT = 0.20  # twenty percent of the remaining groups become validation

# mlp settings used throughout tuning and final refitting
MLP_MAX_ITER = 200  # upper limit only convergence may stop training earlier
MLP_BATCH_SIZE = 128
MLP_ACTIVATION = "relu"
MLP_SOLVER = "adam"

# svm search is staged because fitting many svc models on all 3 072 features is expensive
SVM_SCREENING_GROUP_FRACTION = 0.30
SVM_SCREENING_CV_FOLDS = 3
SVM_CONFIRM_TOP_K = 2
SVM_GRID_N_JOBS = -1  # use all logical cpu cores set to 2 if memory becomes a bottleneck

# use the script location as the project root so relative paths stay portable
PROJECT_DIR = Path(__file__).resolve().parent
DEDUP_ZIP = PROJECT_DIR / "archive_deduplicated.zip"
ORIGINAL_ZIP = PROJECT_DIR / "archive.zip"

# prefer the raw archive so the duplicate removal step is reproduced from the original files
if ORIGINAL_ZIP.is_file():
    ZIP_PATH = ORIGINAL_ZIP
    print(
        "Using original archive.zip so the SHA-256 duplicate-removal step "
        "is reproduced from the raw dataset."
    )
elif DEDUP_ZIP.is_file():
    ZIP_PATH = DEDUP_ZIP
    print(
        "WARNING: using archive_deduplicated.zip. Model results can be reproduced, "
        "but the original 73,139-file duplicate audit cannot be reconstructed "
        "because those duplicates are already absent."
    )
else:
    raise FileNotFoundError(
        "Place archive_deduplicated.zip or archive.zip beside this script:\n"
        f"{PROJECT_DIR}"
    )

# keep extracted data separate from generated experiment outputs
DATA_DIR = PROJECT_DIR / f"extracted_{ZIP_PATH.stem}"
OUTPUT_DIR = PROJECT_DIR / "outputs_staged_full_dataset_local"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# dataset loading and indexing


# extraction is skipped on later runs if the image folder already exists
def extract_dataset():
    """Extract the dataset once and reuse the extracted folder on later runs."""
    if not (DATA_DIR / "myData").is_dir():
        print("Extracting dataset...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with ZipFile(ZIP_PATH) as archive:
            archive.extractall(DATA_DIR)
        print("Extraction complete.")
    else:
        print("Using existing extracted data:", DATA_DIR)



def index_unique_images():
    """
    Index all JPG images and remove exact byte-identical duplicates using SHA-256.

    Related, non-identical sequence frames are retained. Their sequence identity is
    stored in the 'group' column and used later by the group-aware split.
    """
    # map numeric class ids to the traffic sign names supplied with the dataset
    labels_df = pd.read_csv(DATA_DIR / "labels.csv")
    class_names = dict(zip(labels_df["ClassId"].astype(int), labels_df["Name"]))  # class id to human readable sign name

    image_dir = DATA_DIR / "myData"
    records = []
    duplicate_records = []
    seen_hashes = {}

    # sort folders numerically so class processing order is stable between runs
    class_dirs = sorted(
        [p for p in image_dir.iterdir() if p.is_dir()],
        key=lambda p: int(p.name),
    )

    for class_dir in class_dirs:
        class_id = int(class_dir.name)

        for image_path in sorted(class_dir.glob("*.jpg")):
            # sha 256 acts as a content fingerprint identical bytes produce the same digest
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

            # a repeated hash means the image bytes are identical to a file already indexed
            if digest in seen_hashes:
                duplicate_records.append(
                    {
                        "class_id": class_id,
                        "kept_file": str(seen_hashes[digest]),
                        "duplicate_file": str(image_path),
                        "sha256": digest,
                    }
                )
                continue

            seen_hashes[digest] = image_path
            sequence_id = image_path.stem.split("_")[0]  # prefix identifies related frames of the same sign sequence

            records.append(
                {
                    "path": str(image_path),
                    "class_id": class_id,
                    "group": f"{class_id}_{sequence_id}",
                    "sha256": digest,
                }
            )

    # convert the indexed records into tables used by the later split and audit steps
    images_df = pd.DataFrame(records)
    duplicate_df = pd.DataFrame(duplicate_records)

    # always save the duplicate audit table even if it is empty
    duplicate_df.to_csv(
        OUTPUT_DIR / "exact_duplicates_skipped_during_indexing.csv",
        index=False,
    )

    # save class counts after exact de duplication
    class_distribution = (
        images_df.groupby("class_id")
        .size()
        .rename("images")
        .reset_index()
    )
    class_distribution["class_name"] = class_distribution["class_id"].map(class_names)
    class_distribution.to_csv(OUTPUT_DIR / "class_distribution_deduplicated.csv", index=False)  # save class imbalance information for the report

    print(
        "Unique exact images:",
        len(images_df),
        "| Exact duplicates skipped:",
        len(duplicate_df),
        "| Groups:",
        images_df["group"].nunique(),
    )

    return images_df, duplicate_df, class_names, class_distribution


# group aware splitting


def make_group_aware_split(images_df):
    """
    Split independently within each class so related sequence frames remain in one
    partition. Nominal group proportions are 64% train, 16% validation, 20% test.
    Actual image proportions differ because sequence groups have unequal sizes.
    """
    # split labels are added to a copy so the original index table remains unchanged
    sample_df = images_df.copy().reset_index(drop=True)
    sample_df["split"] = ""

    for class_id, class_rows in sample_df.groupby("class_id"):
        # sequence groups are the unit being split rather than individual images
        groups = class_rows["group"].unique()

        if len(groups) < 5:
            raise ValueError(
                f"Class {class_id} has too few groups for the requested split."
            )

        # split sequence ids rather than individual images preventing related frames from crossing partitions
        # reserve test groups first so the final test data stays independent from tuning
        development_groups, test_groups = train_test_split(
            groups,
            test_size=TEST_FRACTION,
            random_state=SEED + int(class_id),
        )

        # split the remaining development groups into training and validation groups
        train_groups, validation_groups = train_test_split(
            development_groups,
            test_size=VALIDATION_FRACTION_OF_DEVELOPMENT,
            random_state=SEED + int(class_id),
        )

        for name, selected_groups in (
            ("train", train_groups),
            ("validation", validation_groups),
            ("test", test_groups),
        ):
            mask = (
                (sample_df["class_id"] == class_id)
                & sample_df["group"].isin(selected_groups)
            )
            sample_df.loc[mask, "split"] = name

    if not (sample_df["split"] != "").all():
        raise RuntimeError("At least one image did not receive a split assignment.")

    group_sets = {
        name: set(sample_df.loc[sample_df["split"] == name, "group"])
        for name in ("train", "validation", "test")
    }

    # explicit leakage check the same traffic sign sequence must never appear in two partitions
    if group_sets["train"] & group_sets["validation"]:
        raise RuntimeError("Group leakage between train and validation.")
    if group_sets["train"] & group_sets["test"]:
        raise RuntimeError("Group leakage between train and test.")
    if group_sets["validation"] & group_sets["test"]:
        raise RuntimeError("Group leakage between validation and test.")

    # report both image counts and sequence group counts because group sizes are unequal
    split_summary = (
        sample_df.groupby("split")
        .agg(images=("path", "size"), groups=("group", "nunique"))
        .reindex(["train", "validation", "test"])
    )
    split_summary["image_fraction"] = split_summary["images"] / len(sample_df)
    split_summary["group_fraction"] = split_summary["groups"] / sample_df["group"].nunique()

    print("\nSplit summary:")
    print(split_summary)
    print("PASS: no sequence group appears in more than one split.")

    sample_df.to_csv(OUTPUT_DIR / "sample_and_split.csv", index=False)  # save exact partition assignments for reproducibility
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv")  # save image and group proportions for each partition

    return sample_df, split_summary


# image preprocessing


def prepare_split(sample_df, name):
    """Convert one partition into the numerical feature matrix expected by MLP and SVM."""
    # isolate one saved partition before converting its images to numerical features
    table = sample_df[sample_df["split"] == name].reset_index(drop=True)

    # preallocate the design matrix one row per image and 3072 flattened rgb features
    X = np.empty(
        (len(table), IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3),
        dtype=np.float32,
    )
    y = table["class_id"].to_numpy(dtype=np.int64)  # integer labels are used directly by sklearn classifiers

    started = time.perf_counter()

    for i, row in enumerate(table.itertuples(index=False), start=1):
        image = imread(row.path)  # load one image using the exact path stored in the split table

        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)  # convert grayscale to 3 channel rgb like input

        image = image[:, :, :3]  # ignore any alpha channel so every sample has exactly three channels
        # resize with anti aliasing to reduce high frequency artefacts introduced during downsampling
        image = resize(
            image,
            (*IMAGE_SIZE, 3),
            anti_aliasing=True,
            preserve_range=True,
        )

        X[i - 1] = (image.astype(np.float32) / 255.0).reshape(-1)  # scale pixel values from 0 to 255 into 0 to 1 then flatten

        if i % 5000 == 0 or i == len(table):
            elapsed = time.perf_counter() - started
            print(
                f"{name}: prepared {i}/{len(table)} images "
                f"({100 * i / len(table):.1f}%) in {elapsed / 60:.1f} min"
            )

    return X, y, table


# mlp model selection


def tune_mlp(X_train, y_train, X_validation, y_validation):
    """Evaluate the three MLP configurations reported in the project."""
    # the first pair isolates hidden layer width the second pair isolates learning rate effects
    # use a small targeted search because a large exhaustive mlp grid would require many expensive fits
    settings_list = [
        {"hidden_layer_sizes": (64,), "learning_rate_init": 0.003},
        {"hidden_layer_sizes": (128,), "learning_rate_init": 0.003},
        {"hidden_layer_sizes": (128,), "learning_rate_init": 0.001},
    ]

    results = []

    for i, settings in enumerate(settings_list, start=1):
        print(f"\nMLP {i}/{len(settings_list)}: {settings}")

        # all non tested settings stay fixed so changes can be attributed to width or learning rate
        model = MLPClassifier(
            **settings,
            activation=MLP_ACTIVATION,
            solver=MLP_SOLVER,
            max_iter=MLP_MAX_ITER,
            batch_size=MLP_BATCH_SIZE,
            early_stopping=False,  # keep model selection tied to the separate validation partition
            n_iter_no_change=10,  # stop if training loss stops improving for several iterations
            random_state=SEED,  # keep initial weights reproducible
        )

        started = time.perf_counter()
        model.fit(X_train, y_train)  # learn weights using training data only
        predictions = model.predict(X_validation)  # validation data is used for model selection not weight updates
        fit_seconds = time.perf_counter() - started

        # record the metrics needed to choose the candidate using unseen validation data
        result = {
            "experiment": i,
            "hidden_layer_sizes": str(settings["hidden_layer_sizes"]),
            "learning_rate_init": settings["learning_rate_init"],
            "actual_iterations": int(model.n_iter_),
            "max_iter": int(model.max_iter),
            "hit_max_iterations": bool(model.n_iter_ >= model.max_iter),
            "final_training_loss": float(model.loss_curve_[-1]),
            "validation_accuracy": float(accuracy_score(y_validation, predictions)),
            "validation_macro_f1": float(
                f1_score(y_validation, predictions, average="macro", zero_division=0)
            ),
            "fit_seconds": float(fit_seconds),
        }

        results.append(result)
        print(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "mlp_validation_checkpoint.csv", index=False)

    # macro f1 is the selection criterion so each class contributes equally despite class imbalance
    best_row = results_df.loc[results_df["validation_macro_f1"].idxmax()]
    best_index = int(best_row["experiment"]) - 1
    best_settings = settings_list[best_index]

    print("\nSelected MLP settings:", best_settings)
    return best_settings, results_df


# svm stage 1 screening


def screen_svm(X_train, y_train, train_table):
    """Screen all 9 SVM candidates using 30% of training groups and 3-fold CV."""
    # build a smaller but class balanced set of sequence groups for the first svm screen
    screening_group_ids = []

    for class_id, class_rows in train_table.groupby("class_id"):
        groups = class_rows["group"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(SEED + int(class_id))  # class specific deterministic sampling

        # keep enough groups for three fold cross validation while targeting about thirty percent per class
        n_groups = max(
            min(SVM_SCREENING_CV_FOLDS, len(groups)),
            int(math.ceil(len(groups) * SVM_SCREENING_GROUP_FRACTION)),
        )
        n_groups = min(n_groups, len(groups))

        selected = rng.choice(groups, size=n_groups, replace=False)
        screening_group_ids.extend(selected.tolist())

    screening_group_ids = set(screening_group_ids)
    mask = train_table["group"].isin(screening_group_ids).to_numpy()  # keep complete groups not random individual frames

    # stage one uses only the selected training groups and does not touch validation or test data
    X_screen = X_train[mask]
    y_screen = y_train[mask]
    groups_screen = train_table.loc[mask, "group"].reset_index(drop=True)

    # three kernels and three c values give nine candidate configurations
    parameter_grid = [
        {"C": [0.1, 1, 10], "kernel": ["linear"]},
        {"C": [0.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale"]},  # gamma scale adapts to feature variance
        {
            "C": [0.1, 1, 10],
            "kernel": ["poly"],
            "gamma": ["scale"],
            "degree": [3],  # use a cubic polynomial kernel
        },
    ]

    # stratifiedgroupkfold balances class proportions while keeping sequence groups intact between folds
    # grid search exhaustively checks the defined nine candidates on the screening subset
    search = GridSearchCV(
        SVC(),
        parameter_grid,
        scoring="f1_macro",
        cv=StratifiedGroupKFold(
            n_splits=SVM_SCREENING_CV_FOLDS,
            shuffle=True,
            random_state=SEED,
        ),
        n_jobs=SVM_GRID_N_JOBS,
        refit=False,  # stage one ranks candidates without fitting a final model
        verbose=2,  # print progress because svm fitting can take a long time
        return_train_score=False,  # avoid computing training scores that are not used for selection
    )

    print(
        "\nSVM Stage 1:",
        len(y_screen),
        "images,",
        len(screening_group_ids),
        "groups",
    )

    started = time.perf_counter()
    search.fit(X_screen, y_screen, groups=groups_screen)  # group ids keep related frames in the same fold
    screening_seconds = time.perf_counter() - started

    results_df = (
        pd.DataFrame(search.cv_results_)[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
                "mean_fit_time",
            ]
        ]
        .sort_values(
            ["rank_test_score", "mean_test_score"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    # add report friendly column names while preserving the raw sklearn fields
    results_df["mean_cv_macro_f1"] = results_df["mean_test_score"]
    results_df["cv_std_deviation"] = results_df["std_test_score"]

    results_df.to_csv(
        OUTPUT_DIR / "svm_stage1_screening_checkpoint.csv",
        index=False,
    )

    top_candidates = results_df.head(SVM_CONFIRM_TOP_K)["params"].tolist()  # only the strongest screen results move to stage 2

    print("\nStage-1 top candidates:")
    for i, params in enumerate(top_candidates, 1):
        print(i, params)

    return (
        top_candidates,
        results_df,
        screening_seconds,
        len(y_screen),
        len(screening_group_ids),
    )


# svm stage 2 confirmation


def confirm_svm(
    top_candidates,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Train the two leading Stage-1 candidates on the full initial training set."""
    results = []

    # only the strongest stage one candidates are promoted to the more expensive full training comparison
    for i, params in enumerate(top_candidates, start=1):
        print(f"\nSVM Stage 2 candidate {i}: {params}")

        model = SVC(**params)  # refit each shortlisted configuration using the full initial training partition

        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        predictions = model.predict(X_validation)  # external validation checks which screened candidate generalises better

        # record the metrics needed to compare the shortlisted candidates
        result = {
            "candidate": i,
            "params": str(params),
            "validation_accuracy": float(accuracy_score(y_validation, predictions)),
            "validation_macro_precision": float(
                precision_score(
                    y_validation, predictions, average="macro", zero_division=0
                )
            ),
            "validation_macro_recall": float(
                recall_score(y_validation, predictions, average="macro", zero_division=0)
            ),
            "validation_macro_f1": float(
                f1_score(y_validation, predictions, average="macro", zero_division=0)
            ),
            "fit_seconds": float(fit_seconds),
        }

        results.append(result)
        print(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "svm_stage2_confirmation_checkpoint.csv",
        index=False,
    )

    # macro f1 is the selection criterion so each class contributes equally despite class imbalance
    # stage 2 rather than the smaller screening subset decides the final svm configuration
    best_row = results_df.loc[results_df["validation_macro_f1"].idxmax()]
    best_index = int(best_row["candidate"]) - 1
    chosen_params = top_candidates[best_index]

    print("\nSelected SVM settings:", chosen_params)
    return chosen_params, results_df


# evaluation helpers


# keep the final model comparison in one consistent metric format
def metric_row(name, y_true, predictions):
    """Return the four final report metrics, using macro averaging for class-balanced comparison."""
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
    }



def per_class_table(y_true, predictions, model_name, class_names, ordered_ids):
    # compute one precision recall f1 value per class so class level behaviour can be inspected later
    # compute one set of metrics for every class instead of only an overall average
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=ordered_ids,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_id": ordered_ids,
            "class_name": [class_names[i] for i in ordered_ids],
            f"{model_name}_precision": precision,
            f"{model_name}_recall": recall,
            f"{model_name}_f1": f1,
            "support": support,
        }
    )



def save_confusion_outputs(y_true, predictions, model_key, title):
    """Export raw counts plus a row-normalized confusion matrix and report-ready figure."""
    labels = np.arange(N_CLASSES)
    cm = confusion_matrix(y_true, predictions, labels=labels)  # rows true classes columns predicted classes

    # save raw counts first so the original prediction totals remain available
    pd.DataFrame(cm).to_csv(
        OUTPUT_DIR / f"{model_key}_confusion_matrix.csv",
        index=False,
        header=False,
    )

    # row normalisation makes each row sum to 1 so diagonal values can be read as per class recall
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )

    pd.DataFrame(cm_norm).to_csv(
        OUTPUT_DIR / f"{model_key}_confusion_matrix_normalized.csv",
        index=False,
        header=False,
    )

    # use the same colour scale for both models so visual comparisons are fair
    plt.figure(figsize=(11, 9))
    plt.imshow(cm_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Proportion of true class")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)
    plt.xticks(range(N_CLASSES), range(N_CLASSES), rotation=90, fontsize=6)
    plt.yticks(range(N_CLASSES), range(N_CLASSES), fontsize=6)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{model_key}_confusion_matrix_normalized.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return cm, cm_norm


# main experiment


def main():
    overall_started = time.perf_counter()  # track the complete runtime including preprocessing and fitting

    print("Project folder:", PROJECT_DIR)
    print("Using ZIP:", ZIP_PATH)
    print("Outputs:", OUTPUT_DIR)

    # build one reproducible dataset index before any model fitting begins
    extract_dataset()
    images_df, duplicate_df, class_names, class_distribution = index_unique_images()
    sample_df, split_summary = make_group_aware_split(images_df)

    # prepare each partition separately so validation and test data never enter training by accident
    X_train, y_train, train_table = prepare_split(sample_df, "train")
    X_validation, y_validation, validation_table = prepare_split(sample_df, "validation")
    X_test, y_test, test_table = prepare_split(sample_df, "test")

    print("\nPrepared arrays:")
    print("Train:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)

    # model selection

    # choose mlp settings only from training and validation performance
    best_mlp_settings, mlp_results_df = tune_mlp(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    (
        top_svm_candidates,
        svm_screening_results_df,
        svm_screening_seconds,
        svm_screening_images,
        svm_screening_groups,
    ) = screen_svm(X_train, y_train, train_table)

    # stage two decides the final svm settings using the separate validation partition
    chosen_svm_params, svm_confirmation_results_df = confirm_svm(
        top_svm_candidates,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    # final refit using train and validation together

    # once hyperparameters are fixed reuse validation samples for learning to maximise final training data
    X_development = np.vstack([X_train, X_validation])
    y_development = np.concatenate([y_train, y_validation])

    development_fraction = len(y_development) / len(sample_df)  # fraction available for the final model fit

    print(
        "\nFinal development images:",
        len(y_development),
        "/",
        len(sample_df),
        "=",
        round(development_fraction, 6),
    )

    # important correction
    # the final mlp is always permitted the full max iter 200
    # do not reuse the selected tuning model iteration count such as 79 as max iter
    final_mlp = MLPClassifier(
        **best_mlp_settings,
        activation=MLP_ACTIVATION,
        solver=MLP_SOLVER,
        max_iter=MLP_MAX_ITER,
        batch_size=MLP_BATCH_SIZE,
        early_stopping=False,  # final training uses all development samples without an internal validation split
        n_iter_no_change=10,  # allow convergence before the maximum iteration limit
        random_state=SEED,  # reproduce the same weight initialisation when rerun
    )

    started = time.perf_counter()
    final_mlp.fit(X_development, y_development)  # refit from scratch using the selected settings
    final_mlp_fit_seconds = time.perf_counter() - started

    final_svm = SVC(**chosen_svm_params)  # same selected hyperparameters now trained on all development samples
    started = time.perf_counter()
    final_svm.fit(X_development, y_development)  # final svm uses the same combined development data
    final_svm_fit_seconds = time.perf_counter() - started

    # final evaluation on the untouched test set

    # the test set has not influenced tuning these predictions provide the final generalisation estimate
    mlp_test_predictions = final_mlp.predict(X_test)
    svm_test_predictions = final_svm.predict(X_test)

    # save predictions so later plots and checks do not require model inference again
    np.save(OUTPUT_DIR / "mlp_test_predictions.npy", mlp_test_predictions)
    np.save(OUTPUT_DIR / "svm_test_predictions.npy", svm_test_predictions)

    # these are the headline test metrics reported in the final comparison table
    comparison_df = pd.DataFrame(
        [
            metric_row("MLP", y_test, mlp_test_predictions),
            metric_row("SVM", y_test, svm_test_predictions),
        ]
    )

    print("\nFINAL TEST RESULTS")
    print(comparison_df.to_string(index=False))

    # full precision values for reproducibility
    comparison_df.to_csv(OUTPUT_DIR / "test_comparison_corrected.csv", index=False)

    # report friendly rounded table
    comparison_df.round(4).to_csv(
        OUTPUT_DIR / "test_comparison_report_4dp.csv",
        index=False,
    )

    # detailed final mlp row used to document the corrected final fit
    final_mlp_details = {
        **metric_row("MLP_corrected_final_refit", y_test, mlp_test_predictions),
        "actual_iterations": int(final_mlp.n_iter_),
        "max_iter": int(final_mlp.max_iter),
        "final_training_loss": float(final_mlp.loss_),
        "fit_seconds": float(final_mlp_fit_seconds),
        "development_images": int(len(y_development)),
        "test_images": int(len(y_test)),
        "development_fraction": float(development_fraction),
    }
    # keep extra mlp convergence details separate from the common model comparison metrics
    pd.DataFrame([final_mlp_details]).to_csv(
        OUTPUT_DIR / "corrected_mlp_test_metrics.csv",
        index=False,
    )

    # classification reports
    pd.DataFrame(
        classification_report(
            y_test,
            mlp_test_predictions,
            labels=np.arange(N_CLASSES),
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(OUTPUT_DIR / "mlp_classification_report.csv")

    pd.DataFrame(
        classification_report(
            y_test,
            svm_test_predictions,
            labels=np.arange(N_CLASSES),
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(OUTPUT_DIR / "svm_classification_report.csv")

    # correct per class comparison for both final models
    ordered_ids = np.array(sorted(class_names.keys()))  # preserve the dataset class id ordering from 0 to 42
    mlp_class = per_class_table(
        y_test, mlp_test_predictions, "MLP", class_names, ordered_ids
    )
    svm_class = per_class_table(
        y_test, svm_test_predictions, "SVM", class_names, ordered_ids
    )

    # align mlp and svm class rows before calculating the difference in f1 scores
    per_class_comparison = mlp_class.merge(
        svm_class.drop(columns="support"),
        on=["class_id", "class_name"],
    )
    # positive values mean the mlp has higher f1 for that class negative values favour the svm
    per_class_comparison["f1_difference_MLP_minus_SVM"] = (
        per_class_comparison["MLP_f1"] - per_class_comparison["SVM_f1"]
    )
    per_class_comparison.to_csv(
        OUTPUT_DIR / "per_class_model_comparison_corrected.csv",
        index=False,
    )

    # save the largest absolute f1 differences as an optional report aid
    selected_per_class = (
        per_class_comparison.assign(
            absolute_f1_difference=lambda d: d["f1_difference_MLP_minus_SVM"].abs()
        )
        .sort_values("absolute_f1_difference", ascending=False)
        .head(10)
    )
    selected_per_class.to_csv(
        OUTPUT_DIR / "largest_per_class_f1_differences.csv",
        index=False,
    )

    # confusion matrices and report ready blue normalized figures
    # generate matching confusion outputs for the two final models
    save_confusion_outputs(
        y_test,
        mlp_test_predictions,
        "mlp",
        "Normalized Confusion Matrix – Final MLP",
    )
    save_confusion_outputs(
        y_test,
        svm_test_predictions,
        "svm",
        "Normalized Confusion Matrix – Final SVM",
    )

    # save the final models
    # pickle stores the trained estimators so predictions can be regenerated without repeating expensive fitting
    with open(OUTPUT_DIR / "final_mlp_model.pkl", "wb") as f:
        pickle.dump(final_mlp, f)

    with open(OUTPUT_DIR / "final_svm_model.pkl", "wb") as f:
        pickle.dump(final_svm, f)

    # save metadata needed to reproduce the run

    # store the main settings counts timings and final metrics in one machine readable file
    config = {
        "seed": SEED,
        "zip_path": str(ZIP_PATH),
        "image_size": list(IMAGE_SIZE),
        "feature_count": int(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3),
        "indexed_image_files_before_dedup": int(len(images_df) + len(duplicate_df)),
        "unique_images": int(len(images_df)),
        "exact_duplicates_skipped": int(len(duplicate_df)),
        "independent_groups": int(images_df["group"].nunique()),
        "class_count": int(N_CLASSES),
        "class_size_min": int(class_distribution["images"].min()),
        "class_size_max": int(class_distribution["images"].max()),
        "train_images_initial": int(len(y_train)),
        "validation_images": int(len(y_validation)),
        "development_images_final_fit": int(len(y_development)),
        "development_fraction_final_fit": float(development_fraction),
        "test_images": int(len(y_test)),
        "mlp_selected_settings": {
            "hidden_layer_sizes": list(best_mlp_settings["hidden_layer_sizes"]),
            "learning_rate_init": float(best_mlp_settings["learning_rate_init"]),
            "activation": MLP_ACTIVATION,
            "solver": MLP_SOLVER,
            "batch_size": MLP_BATCH_SIZE,
        },
        "final_mlp_actual_iterations": int(final_mlp.n_iter_),
        "final_mlp_max_iter": int(final_mlp.max_iter),
        "final_mlp_training_loss": float(final_mlp.loss_),
        "svm_selected_settings": chosen_svm_params,
        "svm_screening_group_fraction": SVM_SCREENING_GROUP_FRACTION,
        "svm_screening_cv_folds": SVM_SCREENING_CV_FOLDS,
        "svm_screening_images": int(svm_screening_images),
        "svm_screening_groups": int(svm_screening_groups),
        "svm_screening_seconds": float(svm_screening_seconds),
        "final_mlp_fit_seconds": float(final_mlp_fit_seconds),
        "final_svm_fit_seconds": float(final_svm_fit_seconds),
        "overall_run_seconds": float(time.perf_counter() - overall_started),
        "final_test_metrics": comparison_df.to_dict(orient="records"),
    }

    (OUTPUT_DIR / "run_config_final.json").write_text(
        json.dumps(config, indent=2, default=str),
        encoding="utf-8",
    )

    report_summary = f"""COMPSYS 306 Project 1 — final run summary

Indexed image files before exact de-duplication: {len(images_df) + len(duplicate_df)}
Exact duplicate files removed: {len(duplicate_df)}
Unique images used: {len(images_df)}
Training images: {len(y_train)}
Validation images: {len(y_validation)}
Final development images (train + validation): {len(y_development)}
Test images: {len(y_test)}

Selected MLP: {best_mlp_settings}, activation={MLP_ACTIVATION}, solver={MLP_SOLVER}, batch_size={MLP_BATCH_SIZE}
Final MLP actual iterations: {final_mlp.n_iter_}
Final MLP maximum iterations: {final_mlp.max_iter}
Final MLP training loss: {final_mlp.loss_:.6f}

Selected SVM: {chosen_svm_params}

Final test metrics:
{comparison_df.to_string(index=False)}
"""
    (OUTPUT_DIR / "report_run_summary.txt").write_text(
        report_summary, encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("DONE — FINAL REPORT OUTPUTS GENERATED")
    print("=" * 72)
    print("Outputs saved to:", OUTPUT_DIR)
    print("\nKey files:")
    key_files = [
        "sample_and_split.csv",
        "split_summary.csv",
        "mlp_validation_checkpoint.csv",
        "svm_stage1_screening_checkpoint.csv",
        "svm_stage2_confirmation_checkpoint.csv",
        "test_comparison_corrected.csv",
        "corrected_mlp_test_metrics.csv",
        "per_class_model_comparison_corrected.csv",
        "mlp_confusion_matrix.csv",
        "svm_confusion_matrix.csv",
        "mlp_confusion_matrix_normalized.png",
        "svm_confusion_matrix_normalized.png",
        "final_mlp_model.pkl",
        "final_svm_model.pkl",
        "run_config_final.json",
    ]
    for name in key_files:
        print(" -", name)


if __name__ == "__main__":
    main()
