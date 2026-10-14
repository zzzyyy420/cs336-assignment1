import torch

def top_p_sampling(probabilities, top_p=0.9):
    """
    Top-p 核采样
    """
    sort_probabilities,idx = torch.sort(probabilities,dim=-1,descending=True) 
    cumulative_probabilities = torch.cumsum(sort_probabilities,dim=-1)

    threshold = top_p
    mask = cumulative_probabilities > threshold
    sort_probabilities[mask] = 0

    #归一化
    sort_probabilities.div_(sort_probabilities.sum(dim=-1,keepdim=True))

    # 随机选择一个概率大于0的token
    next_token_idx = torch.multinomial(sort_probabilities,1)
    next_token_idx = torch.gather(idx,dim=-1,index=next_token_idx)
    return next_token_idx

def temperature_scaling(logits,temperature=1.0):
    probabilities = torch.softmax(logits[:,-1,:]/temperature,dim=-1)
    return probabilities


def decode_token(input_tokens,model,max_tokens_to_generate,top_p=0.9,temperature=1.0):
    model.eval()

    input_tokens = torch.tensor(input_tokens).unsqueeze(0)
    with torch.no_grad(): 
        for _ in range(max_tokens_to_generate):
            if input_tokens == "<endoftext>":
                break
            logits = model(input_tokens)
            probabilities = temperature_scaling(logits,temperature)
            next_token_idx = top_p_sampling(probabilities,top_p)
            input_tokens = torch.cat([input_tokens,next_token_idx],dim=-1) 
    return input_tokens
    