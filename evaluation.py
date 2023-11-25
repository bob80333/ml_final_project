import torch

def evaluate_r_at_k(model, dataloader, device, ks=None):
    if ks is None:
        ks = [1, 2, 4]
    
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        all_ids = []
        for batch in dataloader:
            english_audio, german_audio, idx = batch
            english_audio = english_audio.to(device)
            german_audio = german_audio.to(device)
            ids = idx.to(device)
            
            all_audio = torch.cat((english_audio, german_audio), dim=0)
            all_ids.append(torch.repeat(ids, 2))
            
            all_embeds = model(all_audio)
            all_embeddings.append(all_embeds)
            
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_ids = torch.cat(all_ids, dim=0)
        
        
        # calculate r@k
        r_values = []
        for k in ks:
            r = 0
            for i in range(0, len(all_embeddings), 2):
                # get the index of the first english audio
                idx = all_ids[i]
                # get the embeddings of the english audio
                english_embeds = all_embeddings[i]
                # get the embeddings of all the german audio
                german_embeds = all_embeddings[i+1:]
                # calculate the cosine similarity between the english audio and all the german audio
                cosine_similarities = torch.nn.functional.cosine_similarity(english_embeds, german_embeds)
                # get the indices of the top k most similar german audio
                top_k = cosine_similarities.topk(k=k, largest=True, sorted=True).indices
                # check if the index of the first english audio is in the top k most similar german audio
                if idx in top_k:
                    r += 1
            r /= len(all_embeddings) / 2
            r_values.append(r)