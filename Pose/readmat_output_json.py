import scipy
import numpy as np
import json

datadir = "/run/media/ash/2TB/Datasets/andriluka14cvpr"
mat_path = datadir + "/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"

decoded1 = scipy.io.loadmat(mat_path, struct_as_record=False)["RELEASE"]

must_be_list_fields = ["annolist", "annorect", "point", "img_train", "single_person", "act", "video_list"]

def generate_dataset_obj(obj):
    if type(obj) == np.ndarray:
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])
        else:
            ret = []
            for i in range(dim):
                ret.append(generate_dataset_obj(obj[i]))

    elif type(obj) == scipy.io.matlab.mio5_params.mat_struct:
        ret = {}
        for field_name in obj._fieldnames:
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in must_be_list_fields and type(field) != list:
                field = [field]
            ret[field_name] = field

    else:
        ret = obj

    if(type(ret) == np.uint8 or type(ret) == np.uint16 or type(ret) == np.int16) :
        ret = int(ret)
    if(type(ret) == np.float64 ) :
        ret = float(ret)
    if(type(ret) == np.str_ ) :
        ret = str(ret)

    return ret

# Convert to dict
dataset_obj = generate_dataset_obj(decoded1)

with open('dataset.json', 'w') as f:
        json.dump(dataset_obj, f, ensure_ascii=False, indent=4)
