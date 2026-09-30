"""Task identifiers and class order shared by training, inference and reporting."""

CLASS_TASKS = {
    "multiclass": ["Low", "Intermediate", "High"],
    "binary5": ["PI <= 5", "PI > 5"],
    "binary10": ["PI <= 10", "PI > 10"],
    "binary30": ["PI < 30", "PI >= 30"],
}
TASKS = [*CLASS_TASKS, "pi", "positive_count", "negative_count"]
TASK_NAMES = tuple(TASKS)
CLASS_NAMES = CLASS_TASKS
MULTICLASS_NAMES = tuple(CLASS_TASKS["multiclass"])
BINARY_TASKS = tuple(task for task in CLASS_TASKS if task != "multiclass")
BINARY_CLASS_NAMES = {task: tuple(CLASS_TASKS[task]) for task in BINARY_TASKS}
BINARY_TICK_LABELS = {
    "binary5": ("≤5%", ">5%"),
    "binary10": ("≤10%", ">10%"),
    "binary30": ("<30%", "≥30%"),
}
CLASSIFICATION_TASKS = frozenset(CLASS_TASKS)
REGRESSION_TASKS = frozenset({"pi", "positive_count", "negative_count"})
DISPLAY_CLASS_NAMES = {
    "multiclass": ("Low", "Intermediate", "High"),
    "binary5": ("PI ≤ 5%", "PI > 5%"),
    "binary10": ("PI ≤ 10%", "PI > 10%"),
    "binary30": ("PI < 30%", "PI ≥ 30%"),
}
