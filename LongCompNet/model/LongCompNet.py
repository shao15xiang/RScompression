#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:54:08 2022

@author: xs
"""

import torch
import torch.nn as nn
import math
from .GDN import GDN
from .bitEstimator import BitEstimator
import time


class MaskedConv2d(nn.Conv2d):
    '''
    clone this function from https://github.com/thekoshkina/learned_image_compression/blob/master/masked_conv.py
    Implementation of the Masked convolution from the paper
    Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders."
    Advances in neural information processing systems. 2016.
    https://arxiv.org/pdf/1606.05328.pdf
    '''

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        # fill_将tensor中所有值都填充为指定value
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        # 这相当于对卷积核进行操作嘛
        self.weight.data *= self.mask
        # 调用maskedconv父类，进行卷积呗
        return super(MaskedConv2d, self).forward(x)


class Entropy(nn.Module):
    def __init__(self, num_filters=128):
        super(Entropy, self).__init__()
        self.maskedconv = MaskedConv2d('A', num_filters, num_filters*2, 5, stride=1, padding=2)
        # torch.nn.init.xavier_normal_(self.maskedconv.weight.data, (math.sqrt(2 / (num_filters + num_filters*2))))
        torch.nn.init.xavier_uniform_(self.maskedconv.weight.data, gain=1)
        torch.nn.init.constant_(self.maskedconv.bias.data, 0.0)
        self.conv1 = nn.Conv2d(num_filters*3, 640, 1, stride=1)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(640, 640, 1, stride=1)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(640, num_filters*9, 1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sigma, y):
        y = self.maskedconv(y)
#        print(y.shape)
        x = torch.cat([y, sigma], dim=1)
        # print(x.shape)
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.conv3(x)
        # print("split_size: ", x.shape[1])
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
            torch.split(x, split_size_or_sections=int(x.shape[1]/9), dim=1)
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
        probs = torch.stack([prob0, prob1, prob2], dim=-1)
        # print("probs shape: ", probs.shape)
        probs = self.softmax(probs)
        # probs = torch.nn.Softmax(dim=-1)(probs)
        means = torch.stack([mean0, mean1, mean2], dim=-1)
        variances = torch.stack([scale0, scale1, scale2], dim=-1)

        return means, variances, probs


    
    
class Dual_Non_Local_Block(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192, mid_channel = 96, out_channel_M = 192, k = 5, stride = 1):
        super(Dual_Non_Local_Block, self).__init__()
        
        self.mid_channel = mid_channel
                    
        self.avgkx1  = nn.AdaptiveAvgPool2d((None, 1))
        self.avg1xk  = nn.AdaptiveAvgPool2d((1, None))
        
        self.convkx1 = nn.Conv2d(out_channel_N, mid_channel, (k, 1), stride=1, padding=(k // 2, 0))
        self.conv1xk = nn.Conv2d(out_channel_N, mid_channel, (1, k), stride=1, padding=(0, k // 2))  
        self.convkxk = nn.Conv2d(out_channel_N, mid_channel, 1, stride=1, padding = 0 // 2)
                
        self.theta = nn.Conv2d(self.mid_channel, self.mid_channel, 1, 1, 0)
        self.phi = nn.Conv2d(self.mid_channel, self.mid_channel, 1, 1, 0)

        self.conv1 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channel, out_channel_M, 1, stride=1, padding=0)
        
    def forward(self, x):
            
        x1 = self.avgkx1(x)
        x2 = self.avg1xk(x)
        
        batch_size = x.size(0)
        
        g_x = self.convkxk(x).view(batch_size, self.mid_channel, -1)
        g_x = g_x.permute(0, 2, 1)   # kk x c
#        print(g_x.shape)
        theta_x = self.convkx1(x1).view(batch_size, self.mid_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)   # k x c
        
        phi_x = self.conv1xk(x2).view(batch_size, self.mid_channel, -1) # c x k
        f1 = torch.matmul(theta_x, phi_x)   # k x k
        
        f_div_C = torch.nn.functional.softmax(f1, dim = -1)
        
        f_div_C = f_div_C.view(batch_size, 1, -1)
        
        W_s = f1.view(batch_size, 1,  *x.size()[2:])  
        W_s = self.conv1(W_s)  # 1 x k x k
        vector_S = torch.nn.functional.sigmoid(W_s) # spatial attention vector
       
        g_x = vector_S * x + x
        g_x =  self.convkxk(g_x).view(batch_size, self.mid_channel, -1)
        g_x = g_x.permute(0, 2, 1)   # kk x c
        
        y = torch.matmul(f_div_C, g_x)  # 1 x c
        y = y.permute(0, 2, 1).contiguous()
        
        y = y.view(batch_size, self.mid_channel, 1, 1)
        W_y = self.conv2(y) # c x 1 x 1
        vector_C = torch.nn.functional.sigmoid(W_y) # channel attention vector
        
        x = vector_C * x  + x
        return x
    
    
class LongRangeConv(nn.Module):
    '''
    LongRangeConv 
    '''
    def __init__(self, out_channel_N=192, mid_channel = 96, out_channel_M = 192, k = 5, stride = 1):
        super(LongRangeConv, self).__init__()
        
        self.shortcut = nn.Conv2d(out_channel_N, out_channel_M, 1, stride= stride, padding = 1 // 2)   
        
        
        self.conv1= nn.Conv2d(out_channel_N, mid_channel, 1, stride= 1, padding = 1 // 2)   
        self.conv2= nn.Conv2d(mid_channel, mid_channel, 3, stride= 2, padding = 3 // 2)   
        
        self.convkx1 = nn.Conv2d(mid_channel, mid_channel, (k, 1), stride=1, padding=(k // 2, 0))
        self.conv1xk = nn.Conv2d(mid_channel, mid_channel, (1, k), stride=1, padding=(0, k // 2))  
        
        self.conv3 = nn.Conv2d(2 * mid_channel, out_channel_M, 1, stride= 1, padding = 1 // 2)   
        self.relu = nn.LeakyReLU(inplace = True)
#        self.dual_non_local = Dual_Non_Local_Block(out_channel_N, mid_channel, out_channel_M, 3)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        k_one =self.convkx1(x2)
        one_k = self.conv1xk(x2)
        k_one_k = torch.cat((k_one, one_k), 1) 
        x3 = self.conv3(k_one_k) + self.shortcut(x)
        return x3 



class DeLongRangeConv(nn.Module):
    '''
    LongRangeConv 
    '''
    def __init__(self, out_channel_N=192, mid_channel = 96, out_channel_M = 192, k = 5, stride = 1):
        super(DeLongRangeConv, self).__init__()
        
        self.mid_channel = mid_channel
        
        self.conv1= nn.Conv2d(out_channel_N, mid_channel, 1, stride= 1, padding = 1 // 2)   
        self.conv2= nn.Conv2d(mid_channel, mid_channel, 3, stride= 1, padding = 3 // 2)   
        
        self.convkx1 = nn.Conv2d(mid_channel, mid_channel, (k, 1), stride=1, padding=(k // 2, 0))
        self.conv1xk = nn.Conv2d(mid_channel, mid_channel, (1, k), stride=1, padding=(0, k // 2))  
        self.conv3 = nn.Conv2d(2 * mid_channel, out_channel_N, 1, stride= 1, padding = 1 // 2)   
        
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride= stride, padding=1, output_padding = stride // 2)
        self.relu = nn.LeakyReLU(inplace = True)
#        self.dual_non_local = Dual_Non_Local_Block(out_channel_N, mid_channel, out_channel_M, 3)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        k_one = self.convkx1(x2)
        one_k = self.conv1xk(x2)
        k_one_k = torch.cat((k_one, one_k), 1)
        k_one_k = self.conv3(k_one_k)
        x3 = self.deconv3(k_one_k + x)   
        return x3 


        
class Synthesis_prior_net(nn.Module):
    '''
    Decode synthesis prior
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, attention = None):
        super(Synthesis_prior_net, self).__init__()
        self.deconv1 = DeLongRangeConv(out_channel_M, out_channel_M // 2, out_channel_N, 5, stride=2)
        self.relu1 = nn.ReLU()
        self.deconv2 = DeLongRangeConv(out_channel_N, out_channel_N // 2, out_channel_N, 5, stride=2)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.Conv2d(out_channel_N,  out_channel_M, 3, stride=1, padding =1)
        self.relu3 = nn.ReLU()
#        self.dual_non_local = HLattention(out_channel_M, out_channel_M // 2, 3)
        self.att = attention(out_channel_M, out_channel_M // 2, out_channel_M)
        
    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x =self.deconv3(x)
        return self.att(x)

    
    
class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, outband = 4, attention = None):
        super(Synthesis_net, self).__init__()
        self.att1 = attention(out_channel_M, out_channel_M // 2, out_channel_M)
        self.deconv1 = DeLongRangeConv(out_channel_M, out_channel_M // 2, out_channel_N, 5, stride=2)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        
        self.deconv2 = DeLongRangeConv(out_channel_N, out_channel_N // 2, out_channel_N, 5, stride=2)
        self.igdn2 = GDN(out_channel_N, inverse=True)      
        
        self.att2 = attention(out_channel_N, out_channel_N // 2, out_channel_N)
        
        self.deconv3 = DeLongRangeConv(out_channel_N, out_channel_N // 2, out_channel_N, 5, stride=2)
        
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = DeLongRangeConv(out_channel_N, out_channel_N // 2, outband, 5, stride=2)
                
    def forward(self, x):
        x = self.att1(x)
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.att2(x)
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x



class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, attention = None):
        super(Analysis_prior_net, self).__init__()
        
        self.conv1 = LongRangeConv(out_channel_M, out_channel_M // 2, out_channel_N, 5, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = LongRangeConv(out_channel_N, out_channel_N // 2, out_channel_N, 5, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channel_N,  out_channel_M, 3, stride=1, padding =1)
        self.att = attention(out_channel_M, out_channel_M // 2, out_channel_M)
    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return self.att(x)
    

class Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, inpband = 4, attention = None):
        super(Analysis_net, self).__init__()
        
        
        self.conv1 = nn.Conv2d(inpband, out_channel_N, 5, stride=2, padding = 2)
        self.gdn1 = GDN(out_channel_N)
        
        self.conv2 = LongRangeConv(out_channel_N, out_channel_N // 2, out_channel_N, 5, stride=2)
        self.gdn2 = GDN(out_channel_N)
        
        self.att1 = attention(out_channel_N, out_channel_N // 2, out_channel_N)
        
        self.conv3 = LongRangeConv(out_channel_N, out_channel_N // 2, out_channel_N, 5, stride=2)
        self.gdn3 = GDN(out_channel_N)
        
        self.conv4 = LongRangeConv(out_channel_N, out_channel_N // 2, out_channel_M, 5, stride=2)
        self.att2 = attention(out_channel_M, out_channel_M // 2, out_channel_M)
        

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.att1(x)
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        return self.att2(x)
    
    
    
class AttLongRangeCompressor(nn.Module):
    def __init__(self, out_channel_N=128, out_channel_M = 192, lamb = 2048, band = 3):
        super(AttLongRangeCompressor, self).__init__()
        
        attention = Dual_Non_Local_Block
            
        self.Encoder = Analysis_net(out_channel_N, out_channel_M, band, attention)
                                   
        self.Decoder = Synthesis_net(out_channel_N, out_channel_M, band, attention)
                                    
        self.priorEncoder = Analysis_prior_net(out_channel_N, out_channel_M, attention)
                                            
        self.priorDecoder = Synthesis_prior_net(out_channel_N, out_channel_M , attention)
                                              
        self.bitEstimator_z = BitEstimator(out_channel_M)
        
        self.entropy = Entropy(out_channel_M)
        
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M
        self.lamb = lamb
        

    def forward(self, input_image):
                
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
                
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        x = self.Encoder(input_image)
        batch_size = x.size()[0]
        
        
        z = self.priorEncoder(x)        
        
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
            
        phi= self.priorDecoder(compressed_z)
        
        if self.training:
            compressed_feature = x  + quant_noise_feature
            
        else:       
            compressed_feature = torch.round(x)
            
        means, sigmas, weights = self.entropy(phi, compressed_feature)
        recon_image = self.Decoder(compressed_feature)
        t3 = time.time()
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss =  self.lamb * torch.mean((recon_image - input_image).pow(2))
        
        im_shape = input_image.size()

        def feature_probs_based_sigma(feature, mu, sigma):
            # mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs
        
        def feature_probs_based_GMM(feature, means, sigmas, weights):
            mean1 = torch.squeeze(means[:,:,:,:,0])
            mean2 = torch.squeeze(means[:,:,:,:,1])
            mean3 = torch.squeeze(means[:,:,:,:,2])
            sigma1 = torch.squeeze(sigmas[:,:,:,:,0])
            sigma2 = torch.squeeze(sigmas[:,:,:,:,1])
            sigma3 = torch.squeeze(sigmas[:,:,:,:,2])

            weight1, weight2, weight3 = torch.squeeze(weights[:,:,:,:,0]), torch.squeeze(weights[:,:,:,:,1]), torch.squeeze(weights[:,:,:,:,2])
            # sigma1, sigma2, sigma3 = sigma1.clamp(1e-10, 1e10), sigma2.clamp(1e-10, 1e10), sigma3.clamp(1e-10, 1e10)
            gaussian1 = torch.distributions.laplace.Laplace(mean1, sigma1)
            gaussian2 = torch.distributions.laplace.Laplace(mean2, sigma2)
            gaussian3 = torch.distributions.laplace.Laplace(mean3, sigma3)
            prob1 = gaussian1.cdf(feature + 0.5) - gaussian1.cdf(feature - 0.5)
            prob2 = gaussian2.cdf(feature + 0.5) - gaussian2.cdf(feature - 0.5)
            prob3 = gaussian3.cdf(feature + 0.5) - gaussian3.cdf(feature - 0.5)

            probs = weight1 * prob1 + weight2 * prob2 + weight3 * prob3
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs
        
        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_GMM(compressed_feature, means, sigmas, weights)
        
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        
        return clipped_recon_image, mse_loss, bpp
    
    
if __name__ == '__main__':
    
    
    model = AttLongRangeCompressor(128, 128).cuda()
    model.eval()
    
        
    inp = torch.rand(9,3,256,256).cuda()    
    out = model(inp)
    print(out[-1])
