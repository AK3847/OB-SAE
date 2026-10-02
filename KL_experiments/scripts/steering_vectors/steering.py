import torch



def subtract_layerwise(
    hs_misaligned,
    hs_aligned,
    key="answer"
):

    return {

        layer:
        hs_misaligned[key][layer]
        -
        hs_aligned[key][layer]

        for layer in hs_aligned[key]

    }




def make_random_like(vector):

    random_vectors={}

    for layer,v in vector.items():

        r=torch.randn_like(v)

        random_vectors[layer] = (
            r/r.norm()
        ) * v.norm()


    return random_vectors