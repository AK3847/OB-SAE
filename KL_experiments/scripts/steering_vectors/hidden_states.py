import torch
from peft import PeftModel

def hs_of(out):
    """
    Extract hidden states from transformer block output.
    Handles:
    tensor
    tuple(tensor, ...)
    """
    return out[0] if isinstance(out, tuple) else out


def get_layers(model):

    m = (
        model.base_model.model
        if isinstance(model, PeftModel)
        else model
    )

    return m.model.layers




def collect_hidden_states(
    df,
    model,
    tokenizer,
    n_layers,
    batch_size=2
):

    q_sums={}
    a_sums={}

    q_counts={i:0 for i in range(n_layers)}
    a_counts={i:0 for i in range(n_layers)}


    layers=get_layers(model)

    acts={}


    handles=[]

    for i in range(n_layers):

        h = layers[i].register_forward_hook(
            lambda m, inp, out, i=i:
                acts.__setitem__(
                    i,
                    hs_of(out).detach()
                )
        )

        handles.append(h)



    for start in range(0,len(df),batch_size):

        batch=df.iloc[start:start+batch_size]


        texts=[]
        q_lens=[]


        for q,a in zip(
            batch.question,
            batch.answer
        ):

            user_text = tokenizer.apply_chat_template(
                [{"role":"user","content":q}],
                tokenize=False,
                add_generation_prompt=True
            )

            full_text = tokenizer.apply_chat_template(
                [
                    {"role":"user","content":q},
                    {"role":"assistant","content":a}
                ],
                tokenize=False
            )


            texts.append(full_text)


            q_lens.append(
                len(
                    tokenizer(
                        user_text
                    )["input_ids"]
                )
            )



        inputs=tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        ).to(model.device)



        with torch.no_grad():
            model(**inputs)



        for layer in range(n_layers):

            h=acts[layer]


            for k,ql in enumerate(q_lens):

                real=torch.where(
                    inputs.attention_mask[k].bool()
                )[0]


                qi=real[:ql]
                ai=real[ql:]


                qv=h[k][qi].float().sum(0)
                av=h[k][ai].float().sum(0)



                q_sums[layer]=(
                    qv if layer not in q_sums
                    else q_sums[layer]+qv
                )


                a_sums[layer]=(
                    av if layer not in a_sums
                    else a_sums[layer]+av
                )


                q_counts[layer]+=len(qi)
                a_counts[layer]+=len(ai)



    for h in handles:
        h.remove()



    return {

        "question":
        {
            i:q_sums[i]/q_counts[i]
            for i in range(n_layers)
        },


        "answer":
        {
            i:a_sums[i]/a_counts[i]
            for i in range(n_layers)
        }
    }