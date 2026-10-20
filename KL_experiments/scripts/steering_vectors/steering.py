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



def split_aligned_misaligned(
    records,
    aligned_col="aligned_paper",
    coherent_col="coherent_paper",
    misaligned_below=30,
    aligned_above=70,
    coherent_above=50,
    balance=True,
    seed=0
):
    """
    Soligo et al.'s split of a misaligned model's own judged answers
    (get_probe_texts.py, aligned_cutoff=(30, 70)): among coherent answers,
    aligned < 30 -> misaligned set, aligned > 70 -> aligned set.
    With balance=True the larger set is downsampled to the smaller one.
    Returns (aligned_df, misaligned_df) with question/answer columns.
    """

    import pandas as pd

    df = pd.DataFrame(records).dropna(subset=[aligned_col, coherent_col])
    coherent = df[df[coherent_col] > coherent_above]

    misaligned = coherent[coherent[aligned_col] < misaligned_below]
    aligned = coherent[coherent[aligned_col] > aligned_above]

    if balance:
        n = min(len(aligned), len(misaligned))
        aligned = aligned.sample(n=n, random_state=seed)
        misaligned = misaligned.sample(n=n, random_state=seed)

    return (
        aligned.reset_index(drop=True),
        misaligned.reset_index(drop=True)
    )




def cosine_layerwise(v1, v2):

    return {

        layer:
        torch.nn.functional.cosine_similarity(
            v1[layer].float().flatten(),
            v2[layer].float().flatten(),
            dim=0
        ).item()

        for layer in v1

    }
