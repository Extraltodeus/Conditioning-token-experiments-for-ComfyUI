I made these nodes for fun soooo it's far from perfect but at least it is entertaining!

# The base ideas are:

- Will the next closest similar token activate the model similarly? I would say yes.
- Can I subtract the encoded next similar tokens to the originaly encoded conditioning to make it more precise? I would say "kinda". I've got better results by only influencing the clip_l model output. To this end I've added a node to merge two conditionings, one ending up in the clip_g part and the other will be the clip_l part.

<sub>This little node right there:</sub>

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/fba9be16-ceda-4420-8b91-1aa8a4dccae0)

- Midjourney makes great images by adding stuff. You type "portrait of a girl" and you get something like if you typed a lot more. Dall-e uses a GPT to extrapolate. Could the clip models themselves be used for this? I'd say yes too but I haven't done the best implementation. My guess would be that the next step towards this goal would be to be able to spot the next similar tokens making the most sense/having more influence on the model. Like "banana" for example has "banan" and "banans" next to it but a bit further you can find "pineapple". So maybe by implementing a way to detect this could make a quick way to have more complex prompts. My own implementation kind of has this effect but du to the amount of "not so good" tokens added I'm not 100% convinced by my method.

# The nodes

## Quick and dirty text encode

It doesn't work. I prompted for the candid photography of an Italian woman in Rome and got a kitten.

BUT it is useful for the other nodes in working ways so bear with me.

<sub>Bear with me....</sub>

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/0c8e97f1-16f9-4f58-84cb-0a5495da985f)


<sub>The node:</sub>

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/5cc84428-5636-4d7c-a77f-d9dc39dc75fa)

<sub>The Italian woman in Rome:</sub>

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/c78c8b18-a582-430f-9eef-b1ac4e415171)


## Conditioning to text

Well that's one of the fun nodes! You can either use the usual conditioning directly or one made from the "quick and dirty" node.

Since it uses the same weights (see next node on how to get them) as the quick and dirty, it will be more accurate to find the "next closest tokens".

Example:

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/43641b09-0dac-4b5d-8014-7230ef6d8813)

"Force clip_l" will make it uses the encoded tokens from the clip vit-l model. Not sure why but I get way more coherent results like this.

Despite using weights from SDXL, these nodes are compatible with SD 1.x


## Encode all the tokens (made for SDXL, best used with the SDXL base model).

You will find this simple workflow in the workflows folder. It is necessary to use the previous nodes.

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/d10fba98-8b05-4349-9d50-e61f56d0a716)


## The piece de la resistance
_No but really I speak french and this is a willingful mistake_

![conditioning_similar_tokens](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/d8623a41-f667-458b-bdbf-448c019abc7c)

### the options:

- limit: how many next closest sets of tokens will be generated.
- full_merging: the merging method for the "full_result" output.
  - concatenate: concat the alts
  - average: I let you guess
  - add_diff: original conditioning + original conditioning - alternative conditioning/total alts, for each alt conditioning. Rescaled to the min/max values of the original conditioning.
  - add_diff_loose_rescale (works way better): original conditioning + original conditioning - alternative conditioning, for each alt conditioning. Rescaled at the end by using the average min/max values.
  - max_abs: select the values that are the furthest from zero among the original and the alternative. If you use this I recommand to do the same with the negative prompt and use a "limit" value of 2 or 3.
  - min_abs: select the values that are the closest to zero.
  - smallest relative distances: select the values having the smallest euclidean distances.
  - combine: combine them like the Comfy node does.
- alts_merging: the merging method for the "alts_conditionings" output.
- attention: how the attention will be handled for the alternative conditionings
- divide_loose_rescale: If you use add_diff_loose_rescale, will divide the difference by how many alts created. I recommand to let it on.
- Print alts: for shit and giggles. Or to fool chatGPT and Dall-e with lengthy nonsensical prompts. GPT3.5 and Dall-e do not reject even the most absurd arrangement of tokens. GPT4 however does not seem to understand.


