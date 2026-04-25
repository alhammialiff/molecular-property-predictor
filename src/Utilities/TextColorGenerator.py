from colorama import init, Fore, Style

class TextColorGenerator:
    """
    Utility class to generate colored text for console output using the colorama library.
    This allows us to easily differentiate outputs from different models or stages of training by color-coding them.
    """
    
    def __init__(self):

        # Initialize colorama
        init(autoreset=True)

        self.Fore = Fore
        self.Style = Style
        self.invokeCounter = 0

        self.colorList = {
            # Fore.CYAN,
            'AttentiveFP': Fore.MAGENTA,
            'DMPNN': Fore.YELLOW,
            'GCN': Fore.GREEN,
            'GAT': Fore.RED,
            # Fore.BLUE,
            # Fore.LIGHTBLACK_EX,
            # Fore.LIGHTCYAN_EX,
            # Fore.LIGHTMAGENTA_EX,
            # Fore.LIGHTYELLOW_EX,
            # Fore.LIGHTGREEN_EX,
        }

    def getColour(self, modelName):

        return self.colorList.get(modelName, Fore.WHITE)