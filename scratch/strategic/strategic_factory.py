class StrategicFactory():
    '''
    A class responsible for returning avalanche Strategics.
    '''

    @staticmethod
    def init_plugins(plugins: list, name : str):

        # Inicia os plugins especificados em exp_parser
        p = []
        mem_size = 12 * 16
        if "wamdf" in plugins:
            from . import WAMDFPlugin
            p.append(WAMDFPlugin())
        if "waadb" in plugins:
            from . import WAADBPlugin
            p.append(WAADBPlugin())
        if "replay" in plugins:
            from avalanche.training.plugins import ReplayPlugin
            if name == "UCIHAR_TI":
                mem_size = 4 * 6
            if name == "PAMAP_TI":
                mem_size = 10 * 6
            if name == "DSADS_TI":
                mem_size = 16 * 6
            if name == "HAPT_TI":
                mem_size = 10 * 6
            p.append(ReplayPlugin(mem_size = mem_size)) # TODO: Resolver isso, colocar opção de instanciar automaticamente
        
        return p
