I made these nodes for fun soooo it's far from perfect but at least it is entertaining!

The base ideas were:

- Will the next closest similar token activate the model similarly? I would say yes.
- Can I subtract the encoded next similar tokens to the originaly encoded conditioning to make it more precise? I would say "kinda". I've got better results by only influencing the clip_l model output.
- Midjourney makes great images by adding stuff. You type "portrait of a girl" and you get something like if you typed a lot more. Dall-e uses a GPT to extrapolate. Could the clip models themselves be used for this? I'd say yes too but I haven't done the best implementation. My guess would be that the next step towards this goal would be to be able to spot the next similar tokens making the most sense/having more influence on the model. Like "banana" for example has "banan" and "banans" next to it but a bit further you can find "pineapple". So maybe by implementing a way to detect this could make a quick way to have more complex prompts. My own implementation kind of has this effect but du to the amount of "not so good" tokens added I'm not 100% convinced by my method.

# The
![conditioning_similar_tokens](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/d8623a41-f667-458b-bdbf-448c019abc7c)
