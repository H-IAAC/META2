class StrategicFactory():
    '''
    A class responsible for returning avalanche Strategics.
    '''

    @staticmethod
    def init_plugins(plugins: list):

        # Inicia os plugins especificados em exp_parser
        p = []
        
        if "wamdf" in plugins:
            from . import WAMDFPlugin
            p.append(WAMDFPlugin())
        if "waadb" in plugins:
            from . import WAADBPlugin
            p.append(WAADBPlugin())
        if "replay" in plugins:
            from avalanche.training.plugins import ReplayPlugin
            p.append(ReplayPlugin(mem_size= 12 * 16)) # TODO: Resolver isso, colocar opção de instanciar automaticamente
        
        return p
