# Detail TTS

The model newly proposed three significant important methods to become the best practice of AR TTS. 
- Although RVQ is used, the actual training employs continuous features, I call it fake discretization. 
- All in one model. The model contains gpt, diffusion, vqvae, gan and flowvae all in one. One train one inference.
- Both prefixed spk emb and prompt are used to get benefit from both Valle type inference and Tortoise type training.



NAR version please check ttts.

SVC version please check detail-vc.