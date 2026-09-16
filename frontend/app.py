import streamlit as st
import requests
import matplotlib.pyplot as plt


def format_size(bytes_str):
    try:
        size = int(bytes_str)
    except (ValueError, TypeError):
        return "Unknown size"
    
    if size >= 1024**2:
        return f"{size / (1024**2):.1f} MB"
    elif size >= 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size} bytes"


st.title("DS Toolkit")
BACKEND_URL = "http://localhost:8000"

tab1, tab2, tab3, tab4, tab5, tab6  = st.tabs(["Dataset", "Stats", "Preprocess", "Correlation", "Data Quality", "Split & Train"])

with tab1:
    st.subheader("Search Kaggle Datasets")

    with st.form(key="search_form"):
        query = st.text_input("Search term", placeholder="e.g. titanic")
        search_clicked = st.form_submit_button("Search")

    if search_clicked:
        with st.spinner("Searching Kaggle..."):
            response = requests.get(f"{BACKEND_URL}/search", params={"query": query})
        results = response.json()
        st.session_state["search_results"] = results
    
    if "search_results" in st.session_state:
        results = st.session_state["search_results"]

        selected_ref = st.selectbox(
            "Select a dataset", 
            options=[d["ref"] for d in results],
            format_func=lambda ref: next(
                f"{d['title']} ({d['ref']}, {format_size(d['size'])})" for d in results if d["ref"] == ref
            ),
        )

        if st.button("Download selected dataset"):
            with st.spinner(f"Downloading {selected_ref}..."):
                download_response = requests.post(f"{BACKEND_URL}/download", params={"dataset_ref": selected_ref})

                if download_response.status_code == 200:
                    result = download_response.json()
                    st.session_state["dataset_ref"] = selected_ref
                    if result["cached"]:
                        st.success(f"{selected_ref} was already cached and is now loaded.")
                    else:  
                        st.success(f"Downloaded {selected_ref}")
                else:
                    st.error(f"Download failed: {download_response.text}")
    
    if "dataset_ref" in st.session_state:
        st.info(f"Currently loaded dataset: {st.session_state['dataset_ref']}")


with tab2:
    st.subheader("Dataset Statistics")
    dataset_ref = st.session_state.get("dataset_ref")

    if not dataset_ref:
        st.warning("No dataset loaded. Please download a dataset first.")

    else:
        st.write(f"Statistics for dataset: {dataset_ref}")
        if (
            "stats_data" not in st.session_state
            or st.session_state.get("stats_dataset_ref") != dataset_ref
        ):
            with st.spinner("Fetching statistics..."):
                stats_response = requests.get(f"{BACKEND_URL}/stats", params={"dataset_ref": dataset_ref})
                if stats_response.status_code == 200:
                    st.session_state["stats_data"] = stats_response.json()
                    st.session_state["stats_dataset_ref"] = dataset_ref
                else:
                    st.error(f"Failed to fetch statistics: {stats_response.text}")
                    st.session_state["stats_data"] = None

        stats_data = st.session_state.get("stats_data")

        if stats_data:
            st.write(f"**Shape:** {stats_data['shape'][0]} rows × {stats_data['shape'][1]} columns")
            st.write("**Columns:**", ", ".join(stats_data["columns"]))
            st.write("**Preview (first 5 rows):**")
            st.dataframe(stats_data["head"])
            st.write("**Missing values per column:**")
            st.dataframe([{"column": col, "missing": count} for col, count in stats_data["null_counts"].items()])
            st.write("**Summary statistics:**")
            st.dataframe(stats_data["summary"])


with tab3:
    st.subheader("Preprocess Dataset")
    dataset_ref = st.session_state.get("dataset_ref")

    if not dataset_ref:
        st.warning("No dataset loaded. Please download a dataset first.")
    else:
        stats_data = st.session_state.get("stats_data")
        if not stats_data:
            st.warning("Load the Stats tab first so column info is available")
        else:
            numeric_columns = []
            categorical_columns = []
            for col in stats_data["columns"]:
                col_summary = stats_data["summary"].get(col, {})
                if col_summary.get("mean") is not None:
                    numeric_columns.append(col)
                else:
                    categorical_columns.append(col)

            with st.form("preprocess_form"):
                impute_strategy = st.selectbox("Imputation strategy", ["none", "mean", "median", "mode", "drop"])

                encode_method = st.selectbox("Encoding method", ["none", "onehot", "label"])
                encode_columns = st.multiselect("Columns to encode", options=categorical_columns)

                scale_method = st.selectbox("Scaling method", ["none", "standard", "minmax"])
                scale_columns = st.multiselect("Columns to scale", options=numeric_columns)

                apply_clicked = st.form_submit_button("Apply Preprocessing")

            if apply_clicked:
                params = {
                    "dataset_ref": dataset_ref,
                    "impute_strategy": impute_strategy,
                    "encode_method": encode_method,
                    "scale_method": scale_method,
                }

                with st.spinner("Applying preprocessing..."):
                    response = requests.get(
                        f"{BACKEND_URL}/preprocess",
                        params=[
                            *params.items(),
                            *[("encode_columns", c) for c in encode_columns],
                            *[("scale_columns", c) for c in scale_columns],
                        ],
                    )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state["preprocess_result"] = result
                    st.success("Preprocessing applied")
                else:
                    st.error(f"Preprocessing failed: {response.text}")

            if "preprocess_result" in st.session_state:
                result = st.session_state["preprocess_result"]
                #col1, col2 = st.columns(2)
                #with col1:
                st.write("**Before:**")
                st.dataframe(result["before_preprocessing"], use_container_width=True)
                #with col2:
                st.write("**After:**")
                st.dataframe(result["after_preprocessing"], use_container_width=True)
        

            if st.button("Prepare CSV Export"):
                export_params = [
                    ("dataset_ref", dataset_ref),
                    ("impute_strategy", impute_strategy),
                    ("encode_method", encode_method),
                    ("scale_method", scale_method),
                    *[("encode_columns",c) for c in encode_columns],
                    *[("scale_columns",c) for c in scale_columns]
                ]

                with st.spinner("Preparing export..."):
                    export_response = requests.get(f"{BACKEND_URL}/export", params=export_params)
                
                if export_response.status_code == 200:
                    st.session_state["export_bytes"] = export_response.content
                else:
                    st.error(f"Export failed: {export_response.text}")
            
            if "export_bytes" in st.session_state:
                st.download_button(
                    label="Download processed_dataset.csv",
                    data=st.session_state["export_bytes"],
                    file_name="processed_dataset.csv",
                    mime="text/csv",
                )


with tab4:
    st.subheader("Correlation Matrix")
    dataset_ref = st.session_state.get("dataset_ref")

    if not dataset_ref:
        st.warning("No dataset loaded. Please download a dataset first.")
    else:
        if st.button("Compute Correlation Matrix"):
            with st.spinner("Computing correlations..."):
                response = requests.get(f"{BACKEND_URL}/correlation_matrix", params={"dataset_ref": dataset_ref})
            
            if response.status_code == 200:
                st.session_state["corr_data"] = response.json()["correlation_matrix"]
            else:
                st.error(f"Failed to compute correlation matrix: {response.text}")
        
        if "corr_data" in st.session_state:
            st.dataframe(st.session_state["corr_data"], use_container_width=True)


with tab5:
    st.subheader("Data Quality")
    st.write("Checks for Duplicates and Outliers")

    dataset_ref = st.session_state.get("dataset_ref")

    if not dataset_ref:
        st.warning("No dataset loaded. Please download a dataset first.")
    else:
        stats_data = st.session_state.get("stats_data")
        if not stats_data:
            st.warning("Load the Stats tab first so column info is available")
        else:
            numeric_columns = [
                col for col in stats_data["columns"]
                if stats_data["summary"].get(col, {}).get("mean") is not None
            ]

            st.markdown("### Duplicate Rows")
            remove_duplicates = st.checkbox("Remove duplicates when checking")
            duplicate_subset = st.multiselect(
                "Columns to compare for duplicates (leave empty to compare all)",
                options=stats_data["columns"],
            )

            if st.button("Check Duplicates"):
                dup_params = [
                    ("dataset_ref", dataset_ref),
                    ("remove", remove_duplicates),
                    *[("subset", c) for c in duplicate_subset],
                ]
                with st.spinner("Checking for Duplicates..."):
                    dup_response = requests.get(f"{BACKEND_URL}/duplicates", params=dup_params)

                if dup_response.status_code == 200:
                    st.session_state["dup_result"] = dup_response.json()
                else:
                    st.error(f"Failed to check for duplicates: {dup_response.text}")

            if "dup_result" in st.session_state:
                dup_result = st.session_state["dup_result"]
                st.write(
                    f"**Duplicate count:** {dup_result['duplicate_count']}  |  "
                    f"**Original shape:** {dup_result['original_shape']}  |  "
                    f"**New shape:** {dup_result['new_shape']}"
                )
                st.write("Before:")
                st.dataframe(dup_result["before_preview"], use_container_width=True)
                st.write("After:")
                st.dataframe(dup_result["after"], use_container_width=True)

            st.divider()

            st.markdown("### Outlier Detection (IQR)")
            outlier_columns = st.multiselect("Columns to check for outliers", options=numeric_columns)

            if st.button("Check Outliers"):
                if not outlier_columns:
                    st.warning("Select at least one column.")
                else:
                    out_params = [
                        ("dataset_ref", dataset_ref),
                        *[("columns", c) for c in outlier_columns],
                    ]
                    with st.spinner("Checking for Outliers..."):
                        out_response = requests.get(f"{BACKEND_URL}/outliers", params=out_params)

                    if out_response.status_code == 200:
                        st.session_state["out_result"] = out_response.json()
                    else:
                        st.error(f"Failed to check for outliers: {out_response.text}")

            if "out_result" in st.session_state:
                out_result = st.session_state["out_result"]
                st.write(f"**Cleaned shape:** {out_result['cleaned_shape']}")
                st.write("Per-column outlier summary:")
                st.dataframe(
                    [
                        {"column": col, **{k: v for k, v in info.items() if k != "outlier_indices"}}
                        for col, info in out_result["outlier_info"].items()
                    ],
                    use_container_width=True,
                )
                st.write("Flagged Outlier Rows:")
                st.dataframe(out_result["flagged_rows_preview"], use_container_width=True)


with tab6:
    st.markdown("### Split Dataset")

    dataset_ref = st.session_state.get("dataset_ref")

    if not dataset_ref:
        st.warning("No dataset loaded. Please download a dataset first.")
    else:
        stats_data = st.session_state.get("stats_data")
        if not stats_data:
            st.warning("Load the Stats tab first so column info is available")
        else:
            numeric_columns = [
                col for col in stats_data["columns"]
                if stats_data["summary"].get(col, {}).get("mean") is not None
            ]
            categorical_columns = [col for col in stats_data["columns"] if col not in numeric_columns]

            train_size = st.slider("Train size", min_value=0.1, max_value=0.9, value=0.7, step=0.05)
            test_size = st.slider("Test size", min_value=0.1, max_value=0.3, value=0.15, step=0.05)
            val_size = st.slider("Validation size", min_value=0.1, max_value=0.3, value=0.15, step=0.05)
            random_state = st.number_input("Random seed", value=69, step=1)

            if st.button("Split Dataset"):              
                split_params = [
                    ("dataset_ref", dataset_ref),
                    ("train_size", train_size),
                    ("val_size", val_size),
                    ("test_size", test_size),
                    ("random_state", random_state)
                ]
                with st.spinner("Splitting the Dataset..."):
                    split_response = requests.get(f"{BACKEND_URL}/split", params=split_params)

                if split_response.status_code == 200:
                    st.session_state["split_result"] = split_response.json()
                else:
                    st.error(f"Failed to check for duplicates: {split_response.text}")

                
            if "split_result" in st.session_state:
                split_result = st.session_state["split_result"]
                st.write(
                    f"**Train shape:** {split_result['train_shape']}  |  "
                    f"**Val shape:** {split_result['val_shape']}  |  "
                    f"**Test shape:** {split_result['test_shape']}"
                )

            st.divider()

            st.markdown("### Train Baseline Model")

            with st.form("Training form"):
                target_column_train = st.selectbox("Target column", options=stats_data["columns"], key="train_target_column")
                train_size_train = st.slider("Train size", min_value=0.1, max_value=0.9, value=0.7, step=0.05, key="train_train_size")
                test_size_train = round(1.0 - train_size_train, 2)
                st.write(f"Test size (auto): {test_size_train}")

                task_type_train = st.selectbox("Task type", ["classification", "regression"], key="train_task_type")
                impute_strategy_train = st.selectbox("Imputation strategy", ["none", "mean", "median", "mode", "drop"], key="train_impute")
                encode_method_train = st.selectbox("Encoding method", ["none", "onehot", "label"], key="train_encode_method")
                encode_columns_train = st.multiselect("Columns to encode", options=categorical_columns, key="train_encode_cols")

                scale_method_train = st.selectbox("Scaling method", ["none", "standard", "minmax"], key="train_scale_method")
                scale_columns_train = st.multiselect("Columns to scale", options=numeric_columns, key="train_scale_cols")

                train_clicked = st.form_submit_button("Train Dataset")
            
            if train_clicked:
                train_params = [
                    ("dataset_ref", dataset_ref),
                    ("target_column", target_column_train),
                    ("task_type", task_type_train),
                    ("impute_strategy", impute_strategy_train),
                    ("encode_method", encode_method_train),
                    ("scale_method", scale_method_train),
                    ("train_size", train_size_train),
                    ("test_size", test_size_train),
                    ("random_state", random_state)
                ]

                with st.spinner(f"Training {task_type_train} model..."):
                    train_response = requests.get(
                            f"{BACKEND_URL}/train", 
                            params=[
                                *train_params,
                                *[("encode_columns", c) for c in encode_columns_train],
                                *[("scale_columns", c) for c in scale_columns_train],  
                            ]
                        )

                if train_response.status_code == 200:
                    st.session_state["train_result"] = train_response.json()
                    st.success("Training complete")
                else:
                    st.error(f"Training failed: {train_response.text}")
                
            if "train_result" in st.session_state:
                result = st.session_state["train_result"]
                st.write(f"**Task type:** {result['task_type']}")

                if result["task_type"] == "classification":
                    st.write(
                        f"Accuracy: {result['accuracy']:.3f} | Precision: {result['precision']:.3f} | "
                        f"Recall: {result['recall']:.3f} | F1: {result['f1_score']:.3f}"
                    )
                    st.write("Confusion matrix:")

                    fig, ax = plt.subplots()
                    cm = result["confusion_matrix"]
                    ax.imshow(cm, cmap="Blues")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_xticks(range(len(cm)))
                    ax.set_yticks(range(len(cm)))
                    for i, row in enumerate(cm):
                        for j, val in enumerate(row):
                            ax.text(j, i, val, ha="center", va="center", color="black")
                    st.pyplot(fig)
                else:
                    st.write(f"MSE: {result['mse']:.3f} | R2 score: {result['r2_score']:.3f}")

                    fig, ax = plt.subplots()
                    ax.plot(result["actual_sample"], label="Actual", marker="o")
                    ax.plot(result["predictions_sample"], label="Predicted", marker="o")
                    ax.set_xlabel("Sample index")
                    ax.set_ylabel("Value")
                    ax.legend()
                    st.pyplot(fig)

                st.write("Feature coefficients:")
                st.dataframe(
                    [{"feature": f, "coefficient": c} for f, c in zip(result["feature_names"], result["coefficients"])],
                    use_container_width=True,
                )
#streamlit run app.py