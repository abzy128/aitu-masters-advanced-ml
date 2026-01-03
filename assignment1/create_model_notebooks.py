"""Simple script to create model notebooks from template."""

import json
import re

# Read the template (Ridge notebook)
with open("notebooks/01_ridge_regression.ipynb", "r") as f:
    template_str = f.read()

# Define models and their configurations
models = [
    {
        "num": "02",
        "name_full": "Lasso Regression",
        "name_short": "Lasso",
        "filename": "02_lasso_regression",
        "import_line": "from sklearn.linear_model import Lasso",
        "model_init": "model = Lasso(alpha=1.0, random_state=42, max_iter=10000)",
        "use_scaling": True,
        "result_file": "02_lasso.json",
    },
    {
        "num": "03",
        "name_full": "Elastic Net",
        "name_short": "ElasticNet",
        "filename": "03_elastic_net",
        "import_line": "from sklearn.linear_model import ElasticNet",
        "model_init": "model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)",
        "use_scaling": True,
        "result_file": "03_elastic_net.json",
    },
    {
        "num": "04",
        "name_full": "KNN Regression",
        "name_short": "KNN",
        "filename": "04_knn_regression",
        "import_line": "from sklearn.neighbors import KNeighborsRegressor",
        "model_init": "model = KNeighborsRegressor(n_neighbors=5)",
        "use_scaling": True,
        "result_file": "04_knn.json",
    },
    {
        "num": "05",
        "name_full": "Extra Trees Regression",
        "name_short": "Extra Trees",
        "filename": "05_extra_trees",
        "import_line": "from sklearn.ensemble import ExtraTreesRegressor",
        "model_init": "model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)",
        "use_scaling": False,
        "result_file": "05_extra_trees.json",
    },
    {
        "num": "06",
        "name_full": "AdaBoost Regression",
        "name_short": "AdaBoost",
        "filename": "06_adaboost",
        "import_line": "from sklearn.ensemble import AdaBoostRegressor",
        "model_init": "model = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42)",
        "use_scaling": False,
        "result_file": "06_adaboost.json",
    },
    {
        "num": "07",
        "name_full": "Gradient Boosting Regression",
        "name_short": "Gradient Boosting",
        "filename": "07_gradient_boosting",
        "import_line": "from sklearn.ensemble import GradientBoostingRegressor",
        "model_init": "model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)",
        "use_scaling": False,
        "result_file": "07_gradient_boosting.json",
    },
    {
        "num": "09",
        "name_full": "CatBoost Regression",
        "name_short": "CatBoost",
        "filename": "09_catboost",
        "import_line": "from catboost import CatBoostRegressor",
        "model_init": "model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=False)",
        "use_scaling": False,
        "result_file": "09_catboost.json",
    },
    {
        "num": "10",
        "name_full": "HistGradientBoosting Regression",
        "name_short": "HistGradient",
        "filename": "10_histgradient_boosting",
        "import_line": "from sklearn.ensemble import HistGradientBoostingRegressor",
        "model_init": "model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42)",
        "use_scaling": False,
        "result_file": "10_histgradient.json",
    },
]

# For each model, create a notebook
for model_config in models:
    # Load template
    notebook_str = template_str

    # Replace Ridge-specific content
    notebook_str = notebook_str.replace(
        "Ridge Regression Model", f"{model_config['name_full']} Model"
    )
    notebook_str = notebook_str.replace(
        "Ridge Regression model", f"{model_config['name_full']} model"
    )
    notebook_str = notebook_str.replace(
        "Ridge Regression:", f"{model_config['name_full']}:"
    )
    notebook_str = notebook_str.replace(
        "RIDGE REGRESSION", model_config["name_short"].upper()
    )
    notebook_str = notebook_str.replace("Ridge regression", model_config["name_full"])
    notebook_str = notebook_str.replace(
        '"Ridge Regression"', f'"{model_config["name_full"]}"'
    )
    notebook_str = notebook_str.replace(
        "from sklearn.linear_model import Ridge", model_config["import_line"]
    )
    notebook_str = notebook_str.replace(
        "model = Ridge(alpha=1.0, random_state=42)", model_config["model_init"]
    )
    notebook_str = notebook_str.replace(
        '"01_ridge.json"', f'"{model_config["result_file"]}"'
    )
    notebook_str = notebook_str.replace(
        "../results/01_ridge.json", f"../results/{model_config['result_file']}"
    )

    # Handle scaling section for non-scaling models
    if not model_config["use_scaling"]:
        # Remove StandardScaler import
        notebook_str = notebook_str.replace(
            '"from sklearn.preprocessing import StandardScaler\\n",\n     ', ""
        )

        # Remove scaling section (cell 3)
        # This is complex, so let's just load the JSON and modify it
        notebook_json = json.loads(notebook_str)

        # Find and remove the scaling cells
        new_cells = []
        skip_next = False
        for i, cell in enumerate(notebook_json["cells"]):
            if "source" in cell and isinstance(cell["source"], list):
                cell_text = "".join(cell["source"])
                if "Feature Scaling" in cell_text or "StandardScaler" in cell_text:
                    skip_next = True
                    continue
                if skip_next and cell.get("cell_type") == "code":
                    skip_next = False
                    continue
            new_cells.append(cell)

        notebook_json["cells"] = new_cells
        notebook_str = json.dumps(notebook_json, indent=1)

    # Write notebook
    output_path = f"notebooks/{model_config['filename']}.ipynb"
    with open(output_path, "w") as f:
        f.write(notebook_str)

    print(f"✓ Created {output_path}")

print("\n✓ All model notebooks created successfully!")
