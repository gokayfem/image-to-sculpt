# Inputs

Every input becomes the same case: a working image (`ref/front.png`), its maps, and views. What changes is
where the front image and the views come from.

## The working image

`i2s.py new` copies the original into `input/` (sha256 in `case.json`), pads it to the nearest aspect ratio the
view generator supports (21:9 … 9:16) with the image's own border colour, and resizes it so the long side is
1024 px (`--long-side`). Padding instead of cropping keeps the whole subject; matching the generator's aspect
ratio means every generated view can share the front camera's lens. A transparent PNG keeps its alpha as the
mask (`ref/alpha_mask.png`) and is composited on light grey.

Write `--subject` yourself after looking at the image: a short noun phrase with the subject's kind and look
("ceramic owl figurine", "chibi astronaut", "stocky cartoon troll"). It goes into every view prompt; a vague
subject ("thing") gets vaguer views.

## Kinds

| kind | front maps | watch out for |
|---|---|---|
| AI image / render | MoGe depth + focal, Marigold normals, BiRefNet mask | the generator's own inconsistencies (asymmetric details) |
| photo of an object | same; MoGe's metric focal length matters most here | reflections and transparent parts (glass, chrome) give wrong normals; shadows on the ground can join the mask |
| cut-out PNG | alpha mask, MoGe + Marigold if a key is set | halos at the alpha edge: the mask is the alpha > 127 |
| drawing / sketch / flat art | `prep --restyle`: a 3D render of it first (keeps pose and silhouette) | the restyled image is the reference from then on; say so |
| busy image | `prep --select "the red dragon"`: SAM-3 mask of that subject | occluders in front of the subject cut holes in it; pick an image without them if you can |
| several photos | `views --import DIR`: they become views; angles from file names or found by registration | different lenses and distances: registration fits scale and offset, not focal length |
| idea only | `imagine OUT.png "idea"` → `new` | the generated image is the reference; describe the subject the same way |

## Subjects it fits and does not

Good: characters, creatures, figurines, busts, toys, statues, products with one closed volume and readable
shading. Gaps the photo shows (between legs, arm and body, fingers) are opened by the silhouette cut.

Hard: thin structures (wires, hair strands, whiskers, antennae thinner than ~1 % of the subject), holes no view
sees through (a teapot handle seen edge-on), transparent or mirror-like materials, several disconnected objects
(the largest one is kept), scenes and rooms (not a single volume). Say so before starting rather than after.

## Masks

`prep` keeps the largest connected blob of the mask and fills its holes. If the subject is in pieces in the image
(a sword held away from the body), the pieces not connected to the main blob are dropped: say so, or use
`--select` with a prompt that covers all of them.

## Without an API key

Offline, the front needs a cut-out (alpha) and the views must be imported. With no views at all the sculpt is
the inflated silhouette fitted to nothing but the front outline: its depth is a guess and the report says so.
`i2s.py demo` runs the full pipeline offline on a synthetic subject.
