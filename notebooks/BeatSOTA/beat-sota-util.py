import importlib
bsc = importlib.import_module("beat-sota-config")

COMPLETE_DISPLAY_INTERVAL = bsc.COMPLETE_DISPLAY_INTERVAL

# COMPLETION DISPLAY
CURRENT_MOD = 1
CURRENT_LENGTH = 1
COMPLETE_DIVISOR  = int(100 / COMPLETE_DISPLAY_INTERVAL)

def set_current_display(element_num):
    global CURRENT_MOD
    global CURRENT_LENGTH
    
    CURRENT_MOD = int(element_num / COMPLETE_DIVISOR) or 1
    CURRENT_LENGTH = element_num
    
def display_progress(idx):
    if idx % CURRENT_MOD == 0:
        print(str(int((idx / CURRENT_LENGTH)*100)) + '%', end=' ')