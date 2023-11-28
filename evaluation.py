import torch

# calculate the recall@k
def evaluate_r_at_k(dataloader, model, device, ks):
    model.eval()
    with torch.no_grad():
        all_german_embeds = []
        all_english_embeds = []
        for english_audio, german_audio, _ in dataloader:
            english_audio = english_audio.to(device)
            german_audio = german_audio.to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                german_embeds = model(german_audio)
                english_embeds = model(english_audio)

            all_german_embeds.append(german_embeds)
            all_english_embeds.append(english_embeds)

        all_german_embeds = torch.cat(all_german_embeds, dim=0)
        all_english_embeds = torch.cat(all_english_embeds, dim=0)

        # calculate r@k
        r_values = []
        for k in ks:
            r = 0
            for i in range(
                0,
                len(all_german_embeds),
            ):
                # get the embeddings of the german audio
                # we retrieve english based on german, 
                # to evaluate the model capability for german->english translation with retrieval of english audio for reference
                german_embed = all_german_embeds[i]
                # calculate the cosine similarity between the english audio and all the german audio
                cosine_similarities = torch.nn.functional.cosine_similarity(
                    all_english_embeds, german_embed
                )
                # get the indices of the top k most similar german audio
                top_k = cosine_similarities.topk(k=k, largest=True, sorted=True).indices
                # check if the index of the first english audio is in the top k most similar german audio
                if i in top_k:
                    r += 1
            r /= len(all_german_embeds) / 2
            r_values.append(r)

        print(f"r@{ks}: {r_values}")
        return r_values
