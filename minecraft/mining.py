from minecraft.utils import TYPE_PICKUP, TYPE_TRANSFORM, KEY, MOVE_ACTS


class Mining(object):
    def __init__(self):
        # map
        self.env_name = 'mining'
        nb_block = [1, 3]
        nb_water = [1, 3]

        # object
        obj_list = []
        # crafting tables
        obj_list.append(dict(
            name='workspace', pickable=False, transformable=False,
            oid=0, max=3))
        obj_list.append(dict(
            name='furnace', pickable=False, transformable=False,
            oid=1, max=3))
        obj_list.append(dict(
            name='jewelershop', pickable=False, transformable=False,
            oid=2, max=1))

        # materials: collectable
        obj_list.append(
            dict(name='wood', pickable=True, transformable=False,
                 oid=3, max=2))
        obj_list.append(
            dict(name='stone', pickable=True, transformable=False,
                 oid=4, max=2))
        obj_list.append(
            dict(name='coal', pickable=True, transformable=False,
                 oid=5, max=2))
        obj_list.append(
            dict(name='ironore', pickable=True, transformable=False,
                 oid=6, max=2))
        obj_list.append(
            dict(name='silverore', pickable=True, transformable=False,
                 oid=7, max=2))
        obj_list.append(
            dict(name='goldore', pickable=True, transformable=False,
                 oid=8, max=2))
        obj_list.append(
            dict(name='diamond', pickable=True, transformable=False,
                 oid=9, max=2))

        # materials: synthesized
        obj_list.append(dict(
            name='stick', pickable=False, transformable=True,
            oid=10))
        obj_list.append(dict(
            name='stonepickaxe', pickable=False, transformable=True,
            oid=11))
        obj_list.append(dict(
            name='iron', pickable=False, transformable=True,
            oid=12))
        obj_list.append(dict(
            name='silver', pickable=False, transformable=True,
            oid=13))
        obj_list.append(dict(
            name='ironpickaxe', pickable=False, transformable=True,
            oid=14))
        obj_list.append(dict(
            name='gold', pickable=False, transformable=True,
            oid=15))

        # target 
        obj_list.append(dict(
            name='earings', pickable=False, transformable=True,
            oid=16))
        obj_list.append(dict(
            name='ring', pickable=False, transformable=True,
            oid=17))
        obj_list.append(dict(
            name='goldware', pickable=False, transformable=True,
            oid=18))
        obj_list.append(dict(
            name='bracelet', pickable=False, transformable=True,
            oid=19))
        obj_list.append(dict(
            name='necklace', pickable=False, transformable=True,
            oid=20))

        for obj in obj_list:
            obj['imgname'] = obj['name']+'.png'

        # operation: pickup (type=0) or transform (type=1)
        operation_list = {
            KEY.PICKUP: dict(name='pickup', oper_type=TYPE_PICKUP, key='0'),
            KEY.TRANSFORM: dict(name='transform', oper_type=TYPE_TRANSFORM, key='1'),
        }
        # item = agent+block+water+objects
        item_name_to_iid = dict()
        item_name_to_iid['agent'] = 0
        item_name_to_iid['block'] = 1
        item_name_to_iid['water'] = 2
        for obj in obj_list:
            item_name_to_iid[obj['name']] = obj['oid'] + 3

        # subtask
        subtask_list = []
        subtask_list.append(dict(name='Get wood',   param=(KEY.PICKUP, [], [3], [3]))) #0
        subtask_list.append(dict(name="Get stone",  param=(KEY.PICKUP, [], [4], [4]))) #1
        subtask_list.append(dict(name="Make stick",    param=(KEY.TRANSFORM, [3], [0], [10]))) #2
        subtask_list.append(dict(name="Make stone pickaxe", param=(KEY.TRANSFORM, [10, 4], [0], [11]))) #3
        subtask_list.append(dict(name="Get coal", param=(KEY.PICKUP, [11], [5], [5]))) #4
        subtask_list.append(dict(name="Get iron ore",   param=(KEY.PICKUP, [11], [6], [6]))) #5
        subtask_list.append(dict(name="Get silver ore", param=(KEY.PICKUP, [11], [7], [7]))) #6
        subtask_list.append(dict(name="Make iron",   param=(KEY.TRANSFORM, [5, 6], [1], [12]))) #7
        subtask_list.append(dict(name="Make silver", param=(KEY.TRANSFORM, [5, 7], [1], [13]))) #8
        subtask_list.append(dict(name="Make iron pickaxe", param=(KEY.TRANSFORM, [10, 12], [0], [14]))) #9  
        subtask_list.append(dict(name="Get gold ore",   param=(KEY.PICKUP, [14], [8], [8]))) #10
        subtask_list.append(dict(name="Make gold",   param=(KEY.TRANSFORM, [5, 8], [1], [15]))) #11
        subtask_list.append(dict(name="Get diamond",   param=(KEY.PICKUP, [14], [9], [9]))) #12
        subtask_list.append(dict(name="Make earrings", param=(KEY.TRANSFORM, [13, 9], [2], [16]))) #13  
        subtask_list.append(dict(name="Make ring", param=(KEY.TRANSFORM, [12, 9], [2], [17]))) #14
        subtask_list.append(dict(name="Make goldware", param=(KEY.TRANSFORM, [15], [2], [18]))) #15
        subtask_list.append(dict(name="Make bracelet", param=(KEY.TRANSFORM, [12, 13, 15], [2], [19]))) #16  
        subtask_list.append(dict(name="Make necklace", param=(KEY.TRANSFORM, [15, 9], [2], [20]))) #17

        #subtask_param_to_id = dict()
        subtask_output_to_id = dict()
        subtask_obj_to_id = dict()
        subtask_param_list = []
        for i in range(len(subtask_list)):
            subtask = subtask_list[i]
            par = subtask['param']
            output = par[3][0]
            obj = par[2][0]
            subtask_param_list.append(par)
            #subtask_param_to_id[par] = i
            subtask_output_to_id[output] = i
            if obj in subtask_obj_to_id.keys():
                subtask_obj_to_id[obj].append(i)
            else:
                subtask_obj_to_id[obj]=[i]

        nb_map_obj_type = 0
        for obj in obj_list:
            if 'max' in obj.keys():
                nb_map_obj_type += 1 
        nb_obj_type = len(obj_list)
        nb_operation_type = len(operation_list)

        self.operation_list = operation_list
        self.legal_actions = MOVE_ACTS | {
            KEY.PICKUP, KEY.TRANSFORM}

        self.nb_operation_type = nb_operation_type

        self.object_param_list = obj_list
        self.nb_map_obj_type = nb_map_obj_type
        self.nb_obj_type = nb_obj_type
        self.item_name_to_iid = item_name_to_iid
        self.nb_block = nb_block
        self.nb_water = nb_water
        self.subtask_list = subtask_list
        self.subtask_param_list = subtask_param_list
        #self.subtask_param_to_id = subtask_param_to_id
        self.subtask_output_to_id = subtask_output_to_id
        self.subtask_obj_to_id = subtask_obj_to_id

        self.nb_subtask_type = len(subtask_list)
        self.width = 10
        self.height = 10
        #self.feat_dim = 3*len(subtask_list)+1
        self.ranksep = "0.1"
