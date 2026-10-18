class Config:
    drop_cols = ["customer_id"]
    target = "accepted"
    cat_cols = [
        "gender", "region", "family_status",
        "education", "employment_type", "smartphone_brand"
    ]
    fold_col = "fold"
    fold_index = -1  # TODO: check index up
    submit_cols = ["customer_id", "accepted"]

    seed = 42
