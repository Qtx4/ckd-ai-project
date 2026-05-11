def get_stage(egfr):
    if egfr >= 90:
        return "Stage 1"
    elif egfr >= 60:
        return "Stage 2"
    elif egfr >= 45:
        return "Stage 3a"
    elif egfr >= 30:
        return "Stage 3b"
    elif egfr >= 15:
        return "Stage 4"
    else:
        return "Stage 5"