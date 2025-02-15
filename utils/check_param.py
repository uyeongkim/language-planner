OBJECTS = [
    'AlarmClock',
    'Apple',
    'ArmChair',
    'BaseballBat',
    'BasketBall',
    'Bathtub',
    'BathtubBasin',
    'Bed',
    'Blinds',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Chair',
    'Cloth',
    'CoffeeMachine',
    'CounterTop',
    'CreditCard',
    'Cup',
    'Curtains',
    'Desk',
    'DeskLamp',
    'DishSponge',
    'Drawer',
    'Dresser',
    'Egg',
    'FloorLamp',
    'Footstool',
    'Fork',
    'Fridge',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'HousePlant',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'LaundryHamper',
    'LaundryHamperLid',
    'Lettuce',
    'LightSwitch',
    'Microwave',
    'Mirror',
    'Mug',
    'Newspaper',
    'Ottoman',
    'Painting',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Poster',
    'Pot',
    'Potato',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Shelf',
    'ShowerDoor',
    'ShowerGlass',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Sofa',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'StoveBurner',
    'StoveKnob',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'TeddyBear',
    'Television',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'Toilet',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'Towel',
    'TowelHolder',
    'TVStand',
    'Vase',
    'Watch',
    'WateringCan',
    'Window',
    'WineBottle',
]

MOVABLE_RECEPTACLES = [
    'Bowl',
    'Box',
    'Cup',
    'Mug',
    'Plate',
    'Pan',
    'Pot',
]

RECEPTACLES = {
        'BathtubBasin',
        'Bowl',
        'Cup',
        'Drawer',
        'Mug',
        'Plate',
        'Shelf',
        'SinkBasin',
        'Box',
        'Cabinet',
        'CoffeeMachine',
        'CounterTop',
        'Fridge',
        'GarbageCan',
        'HandTowelHolder',
        'Microwave',
        'PaintingHanger',
        'Pan',
        'Pot',
        'StoveBurner',
        'DiningTable',
        'CoffeeTable',
        'SideTable',
        'ToiletPaperHanger',
        'TowelHolder',
        'Safe',
        'BathtubBasin',
        'ArmChair',
        'Toilet',
        'Sofa',
        'Ottoman',
        'Dresser',
        'LaundryHamper',
        'Desk',
        'Bed',
        'Cart',
        'TVStand',
        'Toaster',
    }

VAL_RECEPTACLE_OBJECTS = {
    'Pot': {'Apple',
            'AppleSliced',
            'ButterKnife',
            'DishSponge',
            'Egg',
            'Fork',
            'Knife',
            'Ladle',
            'Lettuce',
            'LettuceSliced',
            'Potato',
            'PotatoSliced',
            'Spatula',
            'Spoon',
            'Tomato',
            'TomatoSliced'},
    'Pan': {'Apple',
            'AppleSliced',
            'ButterKnife',
            'DishSponge',
            'Egg',
            'Fork',
            'Knife',
            'Ladle',
            'Lettuce',
            'LettuceSliced',
            'Potato',
            'PotatoSliced',
            'Spatula',
            'Spoon',
            'Tomato',
            'TomatoSliced'},
    'Bowl': {'Apple',
             'AppleSliced',
             'ButterKnife',
             'DishSponge',
             'Egg',
             'Fork',
             'Knife',
             'Ladle',
             'Lettuce',
             'LettuceSliced',
             'Potato',
             'PotatoSliced',
             'Spatula',
             'Spoon',
             'Tomato',
             'TomatoSliced',
             'Candle',
             'CD',
             'CellPhone',
             'Cloth',
             'CreditCard',
             'DishSponge',
             'KeyChain',
             'Mug',
             'PaperTowel',
             'Pen',
             'Pencil',
             'RemoteControl',
             'Watch'},
    'CoffeeMachine': {'Mug'},
    'Microwave': {'Apple',
                  'AppleSliced',
                  'Bowl',
                  'Bread',
                  'BreadSliced',
                  'Cup',
                  'Egg',
                  'Glassbottle',
                  'Mug',
                  'Plate',
                  'Potato',
                  'PotatoSliced',
                  'Tomato',
                  'TomatoSliced'},
    'StoveBurner': {'Kettle',
                    'Pan',
                    'Pot'},
    'Fridge': {'Apple',
               'AppleSliced',
               'Bowl',
               'Bread',
               'BreadSliced',
               'Cup',
               'Egg',
               'Glassbottle',
               'Lettuce',
               'LettuceSliced',
               'Mug',
               'Pan',
               'Plate',
               'Pot',
               'Potato',
               'PotatoSliced',
               'Tomato',
               'TomatoSliced',
               'WineBottle'},
    'Mug': {'ButterKnife',
            'Fork',
            'Knife',
            'Pen',
            'Pencil',
            'Spoon',
            'KeyChain',
            'Watch'},
    'Plate': {'Apple',
              'AppleSliced',
              'ButterKnife',
              'DishSponge',
              'Egg',
              'Fork',
              'Knife',
              'Ladle',
              'Lettuce',
              'LettuceSliced',
              'Mug',
              'Potato',
              'PotatoSliced',
              'Spatula',
              'Spoon',
              'Tomato',
              'TomatoSliced',
              'AlarmClock',
              'Book',
              'Candle',
              'CD',
              'CellPhone',
              'Cloth',
              'CreditCard',
              'DishSponge',
              'Glassbottle',
              'KeyChain',
              'Mug',
              'PaperTowel',
              'Pen',
              'Pencil',
              'TissueBox',
              'Watch'},
    'Cup': {'ButterKnife',
            'Fork',
            'Spoon'},
    'Sofa': {'BasketBall',
             'Book',
             'Box',
             'CellPhone',
             'Cloth',
             'CreditCard',
             'KeyChain',
             'Laptop',
             'Newspaper',
             'Pillow',
             'RemoteControl'},
    'ArmChair': {'BasketBall',
                 'Book',
                 'Box',
                 'CellPhone',
                 'Cloth',
                 'CreditCard',
                 'KeyChain',
                 'Laptop',
                 'Newspaper',
                 'Pillow',
                 'RemoteControl'},
    'Box': {'AlarmClock',
            'Book',
            'Candle',
            'CD',
            'CellPhone',
            'Cloth',
            'CreditCard',
            'DishSponge',
            'Glassbottle',
            'KeyChain',
            'Mug',
            'PaperTowel',
            'Pen',
            'Pencil',
            'RemoteControl',
            'Statue',
            'TissueBox',
            'Vase',
            'Watch'},
    'Ottoman': {'BasketBall',
                'Book',
                'Box',
                'CellPhone',
                'Cloth',
                'CreditCard',
                'KeyChain',
                'Laptop',
                'Newspaper',
                'Pillow',
                'RemoteControl'},
    'Dresser': {'AlarmClock',
                'BasketBall',
                'Book',
                'Bowl',
                'Box',
                'Candle',
                'CD',
                'CellPhone',
                'Cloth',
                'CreditCard',
                'Cup',
                'Glassbottle',
                'KeyChain',
                'Laptop',
                'Mug',
                'Newspaper',
                'Pen',
                'Pencil',
                'Plate',
                'RemoteControl',
                'SprayBottle',
                'Statue',
                'TennisRacket',
                'TissueBox',
                'ToiletPaper',
                'ToiletPaperRoll',
                'Vase',
                'Watch',
                'WateringCan',
                'WineBottle'},
    'LaundryHamper': {'Cloth'},
    'Desk': {'AlarmClock',
             'BasketBall',
             'Book',
             'Bowl',
             'Box',
             'Candle',
             'CD',
             'CellPhone',
             'Cloth',
             'CreditCard',
             'Cup',
             'Glassbottle',
             'KeyChain',
             'Laptop',
             'Mug',
             'Newspaper',
             'Pen',
             'Pencil',
             'Plate',
             'RemoteControl',
             'SoapBottle',
             'SprayBottle',
             'Statue',
             'TennisRacket',
             'TissueBox',
             'ToiletPaper',
             'ToiletPaperRoll',
             'Vase',
             'Watch',
             'WateringCan',
             'WineBottle'},
    'Bed': {'BaseballBat',
            'BasketBall',
            'Book',
            'CellPhone',
            'Laptop',
            'Newspaper',
            'Pillow',
            'TennisRacket'},
    'Toilet': {'Candle',
               'Cloth',
               'DishSponge',
               'Newspaper',
               'PaperTowel',
               'SoapBar',
               'SoapBottle',
               'SprayBottle',
               'TissueBox',
               'ToiletPaper',
               'ToiletPaperRoll',
               'HandTowel'},
    'ToiletPaperHanger': {'ToiletPaper',
                          'ToiletPaperRoll'},
    'TowelHolder': {'Towel'},
    'HandTowelHolder': {'HandTowel'},
    'Cart': {'Candle',
             'Cloth',
             'DishSponge',
             'Mug',
             'PaperTowel',
             'Plunger',
             'SoapBar',
             'SoapBottle',
             'SprayBottle',
             'Statue',
             'TissueBox',
             'ToiletPaper',
             'ToiletPaperRoll',
             'Vase',
             'HandTowel'},
    'BathtubBasin': {'Cloth',
                     'DishSponge',
                     'SoapBar',
                     'HandTowel'},
    'SinkBasin': {'Apple',
                  'AppleSliced',
                  'Bowl',
                  'ButterKnife',
                  'Cloth',
                  'Cup',
                  'DishSponge',
                  'Egg',
                  'Glassbottle',
                  'Fork',
                  'Kettle',
                  'Knife',
                  'Ladle',
                  'Lettuce',
                  'LettuceSliced',
                  'Mug',
                  'Pan',
                  'Plate',
                  'Pot',
                  'Potato',
                  'PotatoSliced',
                  'SoapBar',
                  'Spatula',
                  'Spoon',
                  'Tomato',
                  'TomatoSliced',
                  'HandTowel'},
    'Cabinet': {'Book',
                'Bowl',
                'Box',
                'Candle',
                'CD',
                'Cloth',
                'Cup',
                'DishSponge',
                'Glassbottle',
                'Kettle',
                'Ladle',
                'Mug',
                'Newspaper',
                'Pan',
                'PepperShaker',
                'Plate',
                'Plunger',
                'Pot',
                'SaltShaker',
                'SoapBar',
                'SoapBottle',
                'SprayBottle',
                'TissueBox',
                'ToiletPaper',
                'ToiletPaperRoll',
                'Vase',
                'WateringCan',
                'WineBottle',
                'HandTowel'},
    'TableTop': {'AlarmClock',
                 'Apple',
                 'AppleSliced',
                 'BaseballBat',
                 'BasketBall',
                 'Book',
                 'Bowl',
                 'Box',
                 'Bread',
                 'BreadSliced',
                 'ButterKnife',
                 'Candle',
                 'CD',
                 'CellPhone',
                 'Cloth',
                 'CreditCard',
                 'Cup',
                 'DishSponge',
                 'Glassbottle',
                 'Egg',
                 'Fork',
                 'Kettle',
                 'KeyChain',
                 'Knife',
                 'Ladle',
                 'Laptop',
                 'Lettuce',
                 'LettuceSliced',
                 'Mug',
                 'Newspaper',
                 'Pan',
                 'PaperTowel',
                 'Pen',
                 'Pencil',
                 'PepperShaker',
                 'Plate',
                 'Pot',
                 'Potato',
                 'PotatoSliced',
                 'RemoteControl',
                 'SaltShaker',
                 'SoapBar',
                 'SoapBottle',
                 'Spatula',
                 'Spoon',
                 'SprayBottle',
                 'Statue',
                 'TennisRacket',
                 'TissueBox',
                 'ToiletPaper',
                 'ToiletPaperRoll',
                 'Tomato',
                 'TomatoSliced',
                 'Vase',
                 'Watch',
                 'WateringCan',
                 'WineBottle',
                 'HandTowel'},
    'CounterTop': {'AlarmClock',
                   'Apple',
                   'AppleSliced',
                   'BaseballBat',
                   'BasketBall',
                   'Book',
                   'Bowl',
                   'Box',
                   'Bread',
                   'BreadSliced',
                   'ButterKnife',
                   'Candle',
                   'CD',
                   'CellPhone',
                   'Cloth',
                   'CreditCard',
                   'Cup',
                   'DishSponge',
                   'Egg',
                   'Glassbottle',
                   'Fork',
                   'Kettle',
                   'KeyChain',
                   'Knife',
                   'Ladle',
                   'Laptop',
                   'Lettuce',
                   'LettuceSliced',
                   'Mug',
                   'Newspaper',
                   'Pan',
                   'PaperTowel',
                   'Pen',
                   'Pencil',
                   'PepperShaker',
                   'Plate',
                   'Pot',
                   'Potato',
                   'PotatoSliced',
                   'RemoteControl',
                   'SaltShaker',
                   'SoapBar',
                   'SoapBottle',
                   'Spatula',
                   'Spoon',
                   'SprayBottle',
                   'Statue',
                   'TennisRacket',
                   'TissueBox',
                   'ToiletPaper',
                   'ToiletPaperRoll',
                   'Tomato',
                   'TomatoSliced',
                   'Vase',
                   'Watch',
                   'WateringCan',
                   'WineBottle',
                   'HandTowel'},
    'Shelf': {'AlarmClock',
              'Book',
              'Bowl',
              'Box',
              'Candle',
              'CD',
              'CellPhone',
              'Cloth',
              'CreditCard',
              'Cup',
              'DishSponge',
              'Glassbottle',
              'Kettle',
              'KeyChain',
              'Mug',
              'Newspaper',
              'PaperTowel',
              'Pen',
              'Pencil',
              'PepperShaker',
              'Plate',
              'Pot',
              'RemoteControl',
              'SaltShaker',
              'SoapBar',
              'SoapBottle',
              'SprayBottle',
              'Statue',
              'TissueBox',
              'ToiletPaper',
              'ToiletPaperRoll',
              'Vase',
              'Watch',
              'WateringCan',
              'WineBottle',
              'HandTowel'},
    'Drawer': {'Book',
               'ButterKnife',
               'Candle',
               'CD',
               'CellPhone',
               'Cloth',
               'CreditCard',
               'DishSponge',
               'Fork',
               'KeyChain',
               'Knife',
               'Ladle',
               'Newspaper',
               'Pen',
               'Pencil',
               'PepperShaker',
               'RemoteControl',
               'SaltShaker',
               'SoapBar',
               'SoapBottle',
               'Spatula',
               'Spoon',
               'SprayBottle',
               'TissueBox',
               'ToiletPaper',
               'ToiletPaperRoll',
               'Watch',
               'WateringCan',
               'HandTowel'},
    'GarbageCan': {'Apple',
                   'AppleSliced',
                   'Bread',
                   'BreadSliced',
                   'CD',
                   'Cloth',
                   'DishSponge',
                   'Egg',
                   'Lettuce',
                   'LettuceSliced',
                   'Newspaper',
                   'PaperTowel',
                   'Pen',
                   'Pencil',
                   'Potato',
                   'PotatoSliced',
                   'SoapBar',
                   'SoapBottle',
                   'SprayBottle',
                   'TissueBox',
                   'ToiletPaper',
                   'ToiletPaperRoll',
                   'Tomato',
                   'TomatoSliced',
                   'WineBottle',
                   'HandTowel'},
    'Safe': {'CD',
             'CellPhone',
             'CreditCard',
             'KeyChain',
             'Statue',
             'Vase',
             'Watch'},
    'TVStand': {'TissueBox'},
    'Toaster': {'BreadSliced'},
}
VAL_RECEPTACLE_OBJECTS['DiningTable'] = VAL_RECEPTACLE_OBJECTS['TableTop']
VAL_RECEPTACLE_OBJECTS['CoffeeTable'] = VAL_RECEPTACLE_OBJECTS['TableTop']
VAL_RECEPTACLE_OBJECTS['SideTable'] = VAL_RECEPTACLE_OBJECTS['TableTop']
del VAL_RECEPTACLE_OBJECTS['TableTop']

VAL_ACTION_OBJECTS = {
    'Heatable': {'Apple',
                 'AppleSliced',
                 'Bread',
                 'BreadSliced',
                 'Cup',
                 'Egg',
                 'Mug',
                 'Plate',
                 'Potato',
                 'PotatoSliced',
                 'Tomato',
                 'TomatoSliced'},
    'Coolable': {'Apple',
                 'AppleSliced',
                 'Bowl',
                 'Bread',
                 'BreadSliced',
                 'Cup',
                 'Egg',
                 'Lettuce',
                 'LettuceSliced',
                 'Mug',
                 'Pan',
                 'Plate',
                 'Pot',
                 'Potato',
                 'PotatoSliced',
                 'Tomato',
                 'TomatoSliced',
                 'WineBottle'},
    'Cleanable': {'Apple',
                  'AppleSliced',
                  'Bowl',
                  'ButterKnife',
                  'Cloth',
                  'Cup',
                  'DishSponge',
                  'Egg',
                  'Fork',
                  'Kettle',
                  'Knife',
                  'Ladle',
                  'Lettuce',
                  'LettuceSliced',
                  'Mug',
                  'Pan',
                  'Plate',
                  'Pot',
                  'Potato',
                  'PotatoSliced',
                  'SoapBar',
                  'Spatula',
                  'Spoon',
                  'Tomato',
                  'TomatoSliced'},
    'Toggleable': {'DeskLamp',
                   'FloorLamp'},
    'Sliceable': {'Apple',
                  'Bread',
                  'Egg',
                  'Lettuce',
                  'Potato',
                  'Tomato'}
}

def is_correct_object(object):
    return object in OBJECTS or object.strip() == ''

def is_correct_mrecep(object):
    return object in MOVABLE_RECEPTACLES or object.strip() == ''

def is_correct_recep(object):
    return object in RECEPTACLES or object.strip() == ''

def is_correct_recep_relations(object, recep):
    return recep in VAL_RECEPTACLE_OBJECTS and object in VAL_RECEPTACLE_OBJECTS[recep]

def check_wrong_action_relations(object, action):
    if 'heat' in action:
        return object in VAL_ACTION_OBJECTS['Heatable']
    if 'cool' in action:
        return object in VAL_ACTION_OBJECTS['Coolable']
    if 'clean' in action:
        return object in VAL_ACTION_OBJECTS['Cleanable']
    if 'look' in action:
        return object in VAL_ACTION_OBJECTS['Toggleable']
    if 'slice' in action:
        return object in VAL_ACTION_OBJECTS['Sliceable']
    
def check_wrong_pddl_config(pddl):
    if not is_correct_object(pddl['object_target']):
        print('Wrong object target')
        print(pddl['object_target'])
    if not is_correct_object(pddl['parent_target']) or not is_correct_recep(pddl['parent_target']):
        print('Wrong parent target')
        print(pddl['parent_target'])
    if not is_correct_object(pddl['mrecep_target']) or not is_correct_mrecep(pddl['mrecep_target']):
        print('Wrong mrecep target')
        print(pddl['mrecep_target'])
    if pddl['task_type'] not in ['pick_and_place_simple', 'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'pick_clean_then_place_in_recep', 'pick_and_place_with_movable_recep', 'look_at_obj_in_light']:
        print('Wrong task type')
        print(pddl['task_type'])
    if not is_correct_recep_relations(pddl['object_target'], pddl['parent_target'])\
        or (pddl['mrecep_target'].strip() != '' and not is_correct_recep_relations(pddl['object_target'], pddl['mrecep_target']))\
        or (pddl['mrecep_target'].strip() != '' and not is_correct_recep_relations(pddl['mrecep_target'], pddl['parent_target'])):
        print('Wrong recep relation')
        print('{} < {} < {}'.format(pddl['object_target'], pddl['mrecep_target'], pddl['parent_target']))
