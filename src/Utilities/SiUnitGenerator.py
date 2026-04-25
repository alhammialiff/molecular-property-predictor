
class SiUnitGenerator:
    def __init__(self, admetProfilingType=None):

        self.admetProfilingType = admetProfilingType
    
    def generate(self):
        
        match self.admetProfilingType:

            case 'lipophilicity':

                return 'logD at pH 7.4'

            case 'solubility':

                return 'logS (mol/L)'

            case 'toxicity':

                return 'Toxicity (unitless)'

            case _:

                return 'ADMET Property (unitless)'