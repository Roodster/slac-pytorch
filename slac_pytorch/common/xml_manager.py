from xml.etree import ElementTree as ET


class XML:

    def __init__(self):
        pass
    
    
    def print_info(self, tree):
        print("===== density =====")
        
        for elem in tree.iterfind('worldbody/body/geom'):
            print(elem.get('density'))
            
        print("===== friction =====")        
        for elem in tree.iterfind('default/default/geom'):
           print(elem.get('friction'))

    def modify(self, file, values):
        """
        
        Values must have the format:
            "mass": x
        """
        
        
        tree = ET.parse(file)
        self.modify_mass(tree, values['mass'])
        self.modify_friction(tree, values['friction'])
        
        tree.write(file)

    def modify_mass(self, tree, value):
        for elem in tree.iterfind('worldbody/body/geom'):
            elem.set('density', str(value))
            
    def modify_friction(self, tree, value):
        for elem in tree.iterfind('default/default/geom'):
            elem.set('friction', f'{float(value):.1f} .1 .1')