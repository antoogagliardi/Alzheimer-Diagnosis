import torch
import numpy as np

def grad_cam(model_out, model_features, model):
    model_out.backward() # get the gradient of the output with respect to the parameters of the model
    
    grads = []
    # detach the gradient from the model's layers
    for name, param in model.named_parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)

    # pool the gradients across the channels
    pooled_gradients = torch.mean(grads, dim=[-1])

    # weight the channels by corresponding gradients
    copy_features = model_features.clone()
    for i in range(copy_features.shape[1]):
        copy_features[:, i, :, :, :] *= pooled_gradients
    
    #|-- heatmap creation
    # average the channels of the activations
    heatmap = torch.mean(copy_features, dim=0)

    # relu on top of the heatmap (https://arxiv.org/pdf/1610.02391.pdf)
    heatmap = np.maximum(heatmap.detach().numpy(), 0)

    # normalize the heatmap
    heatmap /= torch.max(torch.from_numpy(heatmap))

    return heatmap