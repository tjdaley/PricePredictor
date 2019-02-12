"""
progress_bar.py - a simple console progress bar

From: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
"""
__author__ = "Greenstick"
__version__ = "0.0.1"

class ProgressBar(object):
    """
    Encapsulates a simple console progress bar.
    """
    def __init__(self, iterations:int, prefix:str ='Progress', suffix:str ="Complete", decimals:int =1, length:int = 100, fill:str = '█'):
        """
        Class initializer
        """
        self.iterations = iterations
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill

    def update(self, iteration:int, suffix:None):
        """
        Print an update to the status bar.
        """
        my_suffix = suffix or self.suffix
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.iterations)))
        filled_length = int(self.length * iteration // self.iterations)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print('\r%s |%s| %s%% %s' % (self.prefix, bar, percent, my_suffix), end = '\r')
        
        # Print New Line on Complete
        if iteration >= self.iterations: 
            print()