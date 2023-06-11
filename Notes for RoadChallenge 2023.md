# Notes for RoadChallenge 2023 _Road++_

*[比赛准备专用腾讯会议](https://meeting.tencent.com/dm/KCKgZiiDFBwJ): 858-4744-0040*

### Tasks

##### 8th, June
1. 讨论一下各自负责哪一块
2. 各自负责部分的baseline 代码分析好
3. 查一下相关领域之前的trick
4. 韩师兄提醒：反正还是提醒一下啊，时间不多，大家效率高一点。
5. 韩师兄说：另外这个任务既然是纯2d任务 那应该2080ti能跑的。
6. 分工情况：
  - 比赛：Shelly负责mot赛道，Peter负责event赛道，Alexander负责backbone
  - 代码准备：Shelly负责dataset和mot，Alexander负责network和backbone，Peter负责loss和event


(*讲解*：*MOT这块，主要整理一下他怎么做的，包括网络怎么设计的 head怎么设计的 以及tracker怎么设计的。MOT这块和其他的任务有一点不一样，他evaluate的时候涉及到一个tracker，tracker本身并不训练，但是代码比较复杂 涉及到卡尔曼滤波器啥的。把tracker 和eval的代码整理好。 另外仔细研究下他track1 的评估指标是什么，正常的MOT评估指标是MOTA 之类的 你看看他这里用的是传统指标还是自定义的。如果是自定义的整理清楚他的计算规则。*)

(*讲解*：*Network这块。整理好模型使用的backbone（特征提取网络） 以及模型的通用代码，比如数据增广。这块需要研究baseline代码的地方其实不多，因为我们最后肯定要换掉这块，用上最新的。所以主要精力放到两方面，一是好好研究下最新的大模型backbone 比如eva，internimage 尽量跑通一下这些大模型 然后研究一下怎么和baseline结合。二是去查一下其他比赛比如我昨天发的wad的work report 整理一些通用的提点trick 方便我们后续提点。比如多模型融合，后处理，通用数据增广之类的。包括检测算法的数据增广啥的。event这块，主要集中于baseline代码的实现方式，这是一个新任务，所以一定要把他怎么做的研究清楚。研究清楚他这个网络怎么设计的，怎么预测出event的，怎么评估的。评估event准确率的时候是否涉及到检测的准确性，看看评估指标怎么设计的。这一块因为是新任务，所以我也完全不了解，一定要把代码和评估指标完全搞懂，不然之后的工作就没法做。*)

7. 仔细去读一下比赛的规则，看看比赛允不允许用外部数据训练。

<br>


##### 11th, June
1. 交流代码
2. 交流成果


<br>

### Reference
[积累些打比赛的trick](https://cvpr2023.wad.vision/)
[比赛信息](https://eval.ai/web/challenges/challenge-page/2043/overview)
[比赛评判方式](https://eval.ai/web/challenges/challenge-page/2043/evaluation)
[参考文献引用情况](https://scholar.google.com/scholar?
cites=14076387464732902228&as_sdt=2005&sciodt=0,5&hl=zh-CN)
[MOT相关](https://github.com/ZQPei/deep_sort_pytorch)：这是mot比赛最常用的baseline网络，他好就好在所有的模块都是捏和的，可以随时换成最新的检测器 reid等模块。
[什么是Backbone？](https://www.zhihu.com/question/399611596)
[比赛官方源码](https://github.com/salmank255/ROAD_Waymo_Baseline)
[Baseline是什么意思？](https://www.zhihu.com/question/307805005)
[目标检测之Neck选择](https://zhuanlan.zhihu.com/p/342011052)
[深度学习backbone、neck 和 head介绍](https://zhuanlan.zhihu.com/p/607578342)
[机器学习/深度学习中常见数据集加载（读取）方法](https://blog.csdn.net/gailj/article/details/122142929)

<br>
<br>
<br>


### DETR代码分析

> 代码都是删减版本的，保留整体逻辑，删除一些实现细节

#### 模型

* Backbone

  ```python
  class Joiner(nn.Sequential):
      """
      Backbone is the joiner of ResNet and position embedding
      """
      def __init__(self, backbone, position_embedding):
        	"""
        	Args:
        		backbone: ResNet
        		position_embedding: PositionEmbeddingSine or 
        	"""
          super().__init__(backbone, position_embedding)
  
      def forward(self, tensor_list: NestedTensor):
          xs = self[0](tensor_list)
          out: List[NestedTensor] = []
          pos = []
          for name, x in xs.items():
              out.append(x)
              # position encoding
              pos.append(self[1](x).to(x.tensors.dtype))
  
          return out, pos
    
  ```
  * Backbone由两部分组成，Resnet和PositionEmbedding
  * Resnet用来输出deep feature map
  * PositionEmbedding主要目的是为每一个不同的位置生成一个不同的特征，用来区分空间。
> 思考：需要position embedding吗？传统的transformer需要position embedding是因为全连接是位置无关的，无论这个输入的单词在哪个位置，输出的特征是一样的，但是ResNet因为有padding的存在所以每个位置的feature是不一样的，所以这个position embedding是否需要可以实验下。

* Transormer

  * Transformer
    * transformer 主要分为两部分，encoder和decoder
    * encoder用来编码记忆
    * decoder用来解析结果
    * query embed 是可训练的embedding，每一个query embed对应了一个最终的预测，这个预测可以是bbox也可以是空。猜测，每个output embed隐式的编码了位置信息，每个embed负责一个区域附近的预测。 query embed的数量是和图片中可能存在的最大目标数相关的

  ```python
  def forward(self, src, mask, query_embed, pos_embed):
      # a series of reshape
      ...
      tgt = torch.zeros_like(query_embed)
      
      memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
      hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                        pos=pos_embed, query_pos=query_embed)
      return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
  ```

  * Encoder
    * 每个Encoder是由n个堆叠的相同结构Encoder layer组成的
    * Encoder layer 由两部分组成，一个self attention和一个feed-forward network
    * self attention 用来聚合同一批输入的其他embedding信息
    * feed-forward network 用来提取聚合后的信息
  ```python
  def forward_post(self,
                 src,
                 src_mask: Optional[Tensor] = None,
                 src_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None):
    # src + pos. attach position information to the feature map
    q = k = self.with_pos_embed(src, pos)
    
    src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    # residual connection
    src = src + self.dropout1(src2)
    src = self.norm1(src)
    
    # feed-forward network
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    # residual connection again
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    
    return src
  ```

  * Decoder
    * 同理Decoder是由对应的n个堆叠的Decoder layer组成的
    * Deocder layer 分为三部分
      * 一个self attention 聚集不同的query embedding 信息
      * 一个encoder-decoder attention 从编码的记忆中聚集信息
      * 一个feed-forward network 从聚集来的信息中提取更高层次的信息

  ```python
  def forward_post(self, tgt, memory,
                   tgt_mask: Optional[Tensor] = None,
                   memory_mask: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None,
                   pos: Optional[Tensor] = None,
                   query_pos: Optional[Tensor] = None):
      q = k = self.with_pos_embed(tgt, query_pos)
      tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
      tgt = tgt + self.dropout1(tgt2)
      tgt = self.norm1(tgt)
      
      tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                 key=self.with_pos_embed(memory, pos),
                                 value=memory, attn_mask=memory_mask,
                                 key_padding_mask=memory_key_padding_mask)[0]
      tgt = tgt + self.dropout2(tgt2)
      tgt = self.norm2(tgt)
      
      tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
      tgt = tgt + self.dropout3(tgt2)
      tgt = self.norm3(tgt)
      return tgt
  ```

  * MultiheadAttention
    * 这是transformer的核心代码 用来聚集信息 使用的是pytorch内置的实现
    * self-attention and encoder-decoder-attention都是使用这个类实现的
    * 具体做法是 
      * query, key, value 各自经过一个全连接进行转化
      * query * key 获得匹配关系，经过softmax转化成权重
      * 利用权重对value进行求和，聚集相关的value信息
    * 这里有一个heads参数是对embedding进行分组的，上面的操作是组内进行的 然后最终结果再拼在一起

  ```python
  def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
            need_weights=True, static_kv=False, attn_mask=None):
    """
    Inputs of forward function
        query: [target length, batch size, embed dim]
        key: [sequence length, batch size, embed dim]
        value: [sequence length, batch size, embed dim]
        key_padding_mask: if True, mask padding based on batch size
        incremental_state: if provided, previous time steps are cashed
        need_weights: output attn_output_weights
        static_kv: key and value are static
    Outputs of forward function
        attn_output: [target length, batch size, embed dim]
        attn_output_weights: [batch size, target length, sequence length]
    """
    
    # there was a trick to save computation. I omit them
  	...
    q = self._in_proj_q(query)
    k = self._in_proj_k(key)
    v = self._in_proj_v(value)
              
    # num_heads here is used to split embeddings to different groups like group-wise convolution
    q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
  
    src_len = k.size(1)
                                                                          
    # perform q * k.T to get query key weight                                                                               
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_le
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_l
    
    attn_output_weights = F.softmax(
        attn_output_weights.float(), dim=-1)
   
    attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
    # aggregate values by softmaxed weight                                                    
    attn_output = torch.bmm(attn_output_weights, v)
                                                       
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # another feed-forward network to transform embedding                                                   
    attn_output = self.out_proj(attn_output)
                                                       
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_le
        attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
    else:
        attn_output_weights = None
    return attn_output, attn_output_weights
  ```




* DERT


  * DERT 串起了 backbone，transformer 和生成最后结果的header
  * header 包括class_embed, box_embed

  ```python
  def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
    ...
    self.num_queries = num_queries
    self.transformer = transformer
    
    # transform features to box and class prediction
    self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
    self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
    
    # decode query embedding is trained embeddings implicitly give position information
    self.query_embed = nn.Embedding(num_queries, hidden_dim)
    
    self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
    
    self.backbone = backbone
    self.aux_loss = aux_loss
    
  ```

  ```python
  def forward(self, samples: NestedTensor):
      """ The forward expects a NestedTensor, which consists of:
             - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
             - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
          It returns a dict with the following elements:
             - "pred_logits": the classification logits (including no-object) for all queries.
                              Shape= [batch_size x num_queries x (num_classes + 1)]
             - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                             (center_x, center_y, height, width). These values are normalized in [0, 1],
                             relative to the size of each individual image (disregarding possible padding).
                             See PostProcess for information on how to retrieve the unnormalized bounding box.
             - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                              dictionnaries containing the two above keys for each decoder layer.
      """
      features, pos = self.backbone(samples)
      src, mask = features[-1].decompose()
  
      hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
  
      outputs_class = self.class_embed(hs)
      outputs_coord = self.bbox_embed(hs).sigmoid()
  
      out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
      # aux_loss is the prediction of previous decoder
      if self.aux_loss:
          out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
      return out
  ```

#### 训练

##### pipeline

* train_epoch

  * criterion 用来生成loss

  ```python
  def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                      data_loader: Iterable, optimizer: torch.optim.Optimizer,
                      device: torch.device, epoch: int, max_norm: float = 0):
      model.train()
      criterion.train()
      
      for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
          samples = samples.to(device)
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
  
          outputs = model(samples)
          loss_dict = criterion(outputs, targets)
          weight_dict = criterion.weight_dict
          losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
  
          # reduce losses over all GPUs for logging purposes
          loss_dict_reduced = utils.reduce_dict(loss_dict)
          loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
          loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                      for k, v in loss_dict_reduced.items() if k in weight_dict}
          losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
  
          loss_value = losses_reduced_scaled.item()
  
          optimizer.zero_grad()
          losses.backward()
          if max_norm > 0:
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
          optimizer.step()
  
      # gather the stats from all processes
      return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
  ```

##### label生成

* CocoDetection

  * 这套代码没有在数据上做额外的功夫，直接使用的COCO的annotation，所以重点在loss上

  ```python
  class CocoDetection(torchvision.datasets.CocoDetection):
      def __init__(self, img_folder, ann_file, transforms, return_masks):
          super(CocoDetection, self).__init__(img_folder, ann_file)
          self._transforms = transforms
          self.prepare = ConvertCocoPolysToMask(return_masks)
  
      def __getitem__(self, idx):
          img, target = super(CocoDetection, self).__getitem__(idx)
          image_id = self.ids[idx]
          target = {'image_id': image_id, 'annotations': target}
          img, target = self.prepare(img, target)
          if self._transforms is not None:
              img, target = self._transforms(img, target)
          return img, target
  ```

##### loss定义

* SetCriterion

  * Loss部分的代码也很清楚，主要由两步组成

    * 匹配预测的bbox与gt

    * 根据匹配对计算loss

    * > 思考：我感觉这样有缺陷，因为匈牙利匹配本身不可导，所以如果第一步匹配就偏了那计算出来的loss肯定也是偏的。能否使用nonlocal的方式呢，计算一个权重出来，这种可导的匹配方式即使匹配错误也可以通过优化改正。

  * 匹配对的loss分为两部分 label loss和box loss 

    * label loss是CEloss
    * box loss是L1Loss

  ```python
  """ 
  This class computes the loss for DETR.
  The process happens in two steps:
      1) we compute hungarian assignment between ground truth boxes and the outputs of the model
      2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
  """
  
  def forward(self, outputs, targets):
      """ This performs the loss computation.
      Parameters:
           outputs: dict of tensors, see the output specification of the model for the format
           targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc
      """
      outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
      # Retrieve the matching between the outputs of the last layer and the targets
      # matcher is HungarianMatcher
      indices = self.matcher(outputs_without_aux, targets)
      
      # Compute the average number of target boxes accross all nodes, for normalization purposes
      num_boxes = sum(len(t["labels"]) for t in targets)
      num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
      if is_dist_avail_and_initialized():
          torch.distributed.all_reduce(num_boxes)
      num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
      
      # Compute all the requested losses
      losses = {}
      # self.losses  ['labels', 'boxes', 'cardinality'] and cardinality loss is used only in logging 
      # label loss is CELoss
      # box loss is L1Lossw
      for loss in self.losses:
          losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
      # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
      if 'aux_outputs' in outputs:
          for i, aux_outputs in enumerate(outputs['aux_outputs']):
              indices = self.matcher(aux_outputs, targets)
              for loss in self.losses:
                  if loss == 'masks':
                      # Intermediate masks losses are too costly to compute, we ignore them.
                      continue
                  kwargs = {}
                  if loss == 'labels':
                      # Logging is enabled only for the last layer
                      kwargs = {'log': False}
                  l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                  l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                  losses.update(l_dict)
      return losses
  ```

#### 测试

* evaluation 部分并没有进行特殊的处理，这是在detection等视觉任务中引入transformer最大的一个优点，即end-to-end的输出task目标，而不需要传统方式那样将heat map做后处理然后生成目标。

#### 想法

* Transformer for MOT
  * 我感觉transformer这种end-to-end的方式非常是和把MOT串成一个统一的任务来做。
  * 通过这种方式把MOT统一成一个任务






<br>
<br>
<br>
<br>
<br>

# 对 `2023 Road++` 比赛的准备

## Alexander 做的前期准备：


### 1. 对 `eval_SLOWFAST_R50_ACAR_HR2O.yaml` 文件的解释:
*文件本体*
```yaml
evaluate: True

pretrain:
    path: model_zoo/AVA_SLOWFAST_R50_ACAR_HR2O.pth.tar

result_path: experiments/AVA/eval_SLOWFAST_R50_ACAR_HR2O
manual_seed: 1
print_freq: 20

model:
    freeze_bn: True
    backbone:
        arch: slowfast50
        learnable: True
        kwargs:
            alpha: 4
            beta: 0.125
            fuse_only_conv: False
            fuse_kernel_size: 7
            slow_full_span: True
    neck:
        type: basic
        kwargs:
            bbox_jitter:
                num: 1
                scale: 0.075
            num_classes: 60
            multi_class: True
    head:
        type: acar
        kwargs:
            width: 2304
            roi_spatial: 7
            num_classes: 60
            depth: 2

loss:
    type: ava_criterion
    kwargs:
        pose_softmax: True

val:
    root_path: data
    annotation_path: annotations/ava_val_v2.2_fair_0.85.pkl
    batch_size: 1

    augmentation:
        spatial:
          - type: Scale
            kwargs:
                resize: 256
          - type: ToTensor
            kwargs:
                norm_value: 255.
          - type: Normalize
            kwargs:
                mean: [0.450, 0.450, 0.450]
                std: [0.225, 0.225, 0.225]
        temporal:
            type: TemporalCenterCrop
            kwargs:
                size: 64
                step: 2

    with_label: False
    eval_mAP:
        labelmap: annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt
        groundtruth: annotations/ava_val_v2.2.csv
        exclusions: annotations/ava_val_excluded_timestamps_v2.2.csv

```


其实这是一个YAML格式的配置文件。主要功能是配置用于行动识别的模型以及在验证数据集上评估模型的设置。其中包括：
- 预训练模型的地址
- 结果保存的地址
- 随机数种子
- 每训练多少个batch就输出一次结果
- 模型的架构，包括主干网络(backbone)、neck和头部网络(head)，同时也给出了这些网络的一些超参数设置。主干网络使用慢特征(slowfast)的50层网络，同时learnable为True表示该网络可以进行微调。neck层是普通的全连接层，而头部网络使用的是ACAR网络。
- 定义模型的损失函数类型ava_criterion，并给出了一些配置参数。
- 对于验证集的一些设置，如数据集路径、batch size、数据增强的方式等。其中数据增强主要是针对数据的空间和时间维度做了一些数据增强，如缩放、裁剪、变换等。"""
- 执行评估时是否需要给数据打标签
- 对于验证集上评估的一些设定，如标签映射路径、真实值文件、排除时间戳文件等。

其它的YAML文件在内容上也都大同小异。


### 2. About codes of `make_anchors/base_anchors.py` :
```python
import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np

class anchorBox(object):
    """Compute anchorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, aspect_ratios =[0.5, 1 / 1., 1.5],
                    scale_ratios = [1.,]):
        super(anchorBox, self).__init__()
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.default_sizes= [0.01, 0.06, 0.2, 0.4, 0.85]
        self.anchor_boxes = len(self.aspect_ratios)*len(self.scale_ratios)
        self.ar = self.anchor_boxes
        self.num_anchors = self.ar
        
        print(self.scale_ratios, self.ar)

    def forward(self, grid_sizes):
        anchors = []
        for k, f in enumerate(grid_sizes):
            for i, j in product(range(f), repeat=2):
                f_k = 1 / f
                # unit center x,y
                cx = (j + 0.5) * f_k
                cy = (i + 0.5) * f_k
                s = self.default_sizes[k]
                s *= s
                for ar in self.aspect_ratios:  # w/h = ar
                    h = sqrt(s / ar)
                    w = ar * h
                    for sr in self.scale_ratios:  # scale
                        anchor_h = h * sr
                        anchor_w = w * sr
                        anchors.append([cx, cy, anchor_w, anchor_h])
                        print(cx, cy, anchor_w, anchor_h)
        output = torch.FloatTensor(anchors).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output
```


It defines a class called `anchorBox` that is used to compute anchor box coordinates in center-offset form for each source feature map. Anchor boxes are commonly used in object detection algorithms, particularly in the region-based approaches like Faster R-CNN.

Let's go through the code step by step:

#### (1) The code imports necessary libraries and modules:
   - `torch` is the main library for tensor operations and neural networks.
   - `sqrt` function from `math` module is imported and given an alias `sqrt`.
   - `product` function from `itertools` module is imported and given an alias `product`.
   - `numpy` is imported and given an alias `np`.

#### (2) The `anchorBox` class is defined, which represents a collection of anchor boxes.
   - The `__init__` method is the constructor that initializes the object. It takes two optional arguments: `aspect_ratios` and `scale_ratios`. If not provided, default values are used.
   - `aspect_ratios` is a list of aspect ratios for the anchor boxes. In this code, the default aspect ratios are set as `[0.5, 1 / 1., 1.5]`.
   - `scale_ratios` is a list of scale ratios for the anchor boxes. The default value is `[1.,]`.
   - `default_sizes` is a list of default sizes for the anchor boxes.
   - The total number of anchor boxes is calculated as the product of the number of aspect ratios and the number of scale ratios.
   - The variables `ar` and `num_anchors` are assigned the same value, which represents the total number of anchor boxes.
   - The scale ratios and the number of anchor boxes are printed.

(3) The `forward` method is defined to compute the anchor box coordinates.
   - It takes a single argument `grid_sizes`, which is a list of sizes for each source feature map.
   - The `anchors` list is initialized to store the computed anchor box coordinates.
   - The method iterates over each feature map size (`f`) in the `grid_sizes` list.
   - It then iterates over the grid cells within each feature map using the `product` function.
   - For each grid cell, the center coordinates (`cx` and `cy`) of the anchor box are calculated based on the grid cell indices and the feature map size.
   - The default size (`s`) for the current feature map is calculated and squared.
   - The method further iterates over each aspect ratio (`ar`) in the `aspect_ratios` list.
   - For each aspect ratio, the height (`h`) and width (`w`) of the anchor box are computed based on the default size and aspect ratio.
   - Next, it iterates over each scale ratio (`sr`) in the `scale_ratios` list.
   - For each scale ratio, the scaled height (`anchor_h`) and width (`anchor_w`) of the anchor box are calculated.
   - The anchor box coordinates `[cx, cy, anchor_w, anchor_h]` are appended to the `anchors` list.
   - The computed coordinates are also printed.
   - Finally, the `anchors` list is converted to a `torch.FloatTensor` and reshaped into a 2D tensor with shape (-1, 4), where -1 means the size is inferred based on the other dimensions.
   - The coordinates in the tensor are then clamped between 0 and 1 using the `clamp_` method.
   - The resulting tensor is returned as the output of the `forward` method.

So generally speaking, the single class mentioned in this file provides a mechanism to compute anchor box coordinates for object detection models.






### 2. About codes of `models/backbones/slowfast.py` :
#### (1) Part I:
```python
"""
References:
[SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982),
[PySlowFast](https://github.com/facebookresearch/slowfast).
"""
```
It means that this file references [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982) and [PySlowFast](https://github.com/facebookresearch/slowfast).

#### (2) Part II:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```
It means the author imported necessary modules for this file. Among them are one of the most important library called 'troch', which you have to be careful when installing since Pytorch has to match the corresponding version of GPU drivers and CUDA.

#### (3) Part III:
```python
BN = nn.BatchNorm3d
__all__ = ['slowfast50', 'slowfast101', 'slowfast152', 'slowfast200']
```
The code above is a snippet that imports and assigns values to variables in the context of a deep learning model, specifically related to the SlowFast network architecture. Let's break it down:

[1] `BN = nn.BatchNorm3d`
   - This line assigns the `BatchNorm3d` class from the `nn` module (presumably `torch.nn`) to the variable `BN`.
   - `BatchNorm3d` is a batch normalization layer specifically designed for 3D data, commonly used in video-based deep learning models.
   - By assigning `nn.BatchNorm3d` to `BN`, the code creates an alias for the `BatchNorm3d` class, allowing it to be referenced using the shorthand `BN` elsewhere in the code.

[2] `__all__ = ['slowfast50', 'slowfast101', 'slowfast152', 'slowfast200']`
   - This line assigns a list of strings to the `__all__` variable.
   - `__all__` is a special variable in Python that defines the list of names that should be imported when using the `from module import *` syntax.
   - In this case, the names included in the list are `'slowfast50'`, `'slowfast101'`, `'slowfast152'`, and `'slowfast200'`.
   - These names likely correspond to different variations of the SlowFast network architecture, representing different model sizes or depths.
   - By including these names in `__all__`, the code specifies that when using the `from module import *` syntax (where `module` is the module where this code is present), these specific names will be imported.

It assigns the `BatchNorm3d` class to the variable `BN` and defines the list of names (`slowfast50`, `slowfast101`, `slowfast152`, `slowfast200`) that should be imported when using the `from module import *` syntax. These names correspond to different variations of the SlowFast network architecture.

#### (4) Part IV:
```python
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = BN(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = BN(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), 
            padding=(0, dilation, dilation), dilation=(1, dilation, dilation), bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_bn = BN(planes * 4)
        self.stride = stride
        # self.alpha = 1

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)
            res = self.downsample_bn(res)

        out = out + res
        out = self.relu(out)

        return out
```

This is actually a class named `Bottleneck`, which represents a bottleneck block in a convolutional neural network (CNN). This block is commonly used in architectures like ResNet.

[1] The `Bottleneck` class inherits from the `nn.Module` class, which is a base class for all neural network modules in PyTorch.
[2] The `expansion` class variable is defined with a value of 4. This variable is used to determine the number of output channels in the third convolutional layer. The output channels will be `planes * expansion`.
[3] The `__init__` method is the constructor that initializes the `Bottleneck` object. It takes several arguments:
   - `inplanes` represents the number of input channels to the bottleneck block.
   - `planes` represents the number of output channels after the first convolutional layer.
   - `stride` is the stride used in the second convolutional layer. By default, it is set to 1.
   - `downsample` is an optional argument used for downsampling the input tensor. It can be `None` or a downsampling module.
   - `dilation` is the dilation rate used in the second convolutional layer. By default, it is set to 1.
   - `head_conv` specifies the type of convolution used in the first layer. It can be 1 or 3.
[4] The method begins by checking the value of `head_conv` and initializing the first convolutional layer (`self.conv1`) and the corresponding batch normalization layer (`self.bn1`) accordingly.
   - If `head_conv` is 1, a 1x1x1 convolution is used, followed by batch normalization.
   - If `head_conv` is 3, a 3x1x1 convolution with padding is used, followed by batch normalization.
   - If `head_conv` is neither 1 nor 3, a `ValueError` is raised to indicate an unsupported `head_conv` value.
[5] The second convolutional layer (`self.conv2`) is defined using the specified parameters:
   - The kernel size is (1, 3, 3), indicating a 3D convolution with a 1x3x3 kernel.
   - The stride is set to (1, stride, stride), which allows for spatial downsampling in the spatial dimensions.
   - Padding is applied to maintain the spatial dimensions.
   - The dilation rate is set to (1, dilation, dilation), which controls the spacing between kernel elements.
[6] The corresponding batch normalization layer (`self.bn2`) for the second convolutional layer is defined.
[7] The third convolutional layer (`self.conv3`) is defined with a 1x1x1 kernel size to reduce the number of channels back to `planes * expansion`. This layer is followed by batch normalization (`self.bn3`).
[8] An instance of the ReLU activation function (`self.relu`) is created with the `inplace` parameter set to `True`. This means the activation function operates in-place, saving memory.
[9] The `downsample` module is assigned to `self.downsample` for downsampling the input tensor if it is not `None`. Additionally, a batch normalization layer (`self.downsample_bn`) is created if `downsample` is not `None`.
[10] The `stride` class variable is set based on the provided `stride` argument.
[11] The `forward` method defines the forward pass of the `Bottleneck` block. It takes an input tensor `x` and returns the output tensor.
   - The input tensor `x` is saved as `res` for



#### (5) Part V:
```python
class SlowFast(nn.Module):
    def __init__(self, block, layers, alpha=8, beta=0.125, fuse_only_conv=True, fuse_kernel_size=5, slow_full_span=False):
        super(SlowFast, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.slow_full_span = slow_full_span

        '''Fast Network'''
        self.fast_inplanes = int(64 * beta)
        self.fast_conv1 = nn.Conv3d(3, self.fast_inplanes, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = BN(self.fast_inplanes)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fast_res1 = self._make_layer_fast(block, int(64 * beta), layers[0], head_conv=3)
        self.fast_res2 = self._make_layer_fast(block, int(128 * beta), layers[1], stride=2, head_conv=3)
        self.fast_res3 = self._make_layer_fast(block, int(256 * beta), layers[2], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(block, int(512 * beta), layers[3], head_conv=3, dilation=2)

        '''Slow Network'''
        self.slow_inplanes = 64
        self.slow_conv1 = nn.Conv3d(3, self.slow_inplanes, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = BN(self.slow_inplanes)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.slow_res1 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res2 = self._make_layer_slow(block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res3 = self._make_layer_slow(block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res4 = self._make_layer_slow(block, 512, layers[3], head_conv=3, dilation=2)

        

        '''Lateral Connections'''
        fuse_padding = fuse_kernel_size // 2
        fuse_kwargs = {'kernel_size': (fuse_kernel_size, 1, 1), 'stride': (alpha, 1, 1), 'padding': (fuse_padding, 0, 0), 'bias': False}
        if fuse_only_conv:
            def fuse_func(in_channels, out_channels):
                return nn.Conv3d(in_channels, out_channels, **fuse_kwargs)
        else:
            def fuse_func(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, **fuse_kwargs),
                    BN(out_channels),
                    nn.ReLU(inplace=True)
                )
        self.Tconv1 = fuse_func(int(64 * beta), int(128 * beta))
        self.Tconv2 = fuse_func(int(256 * beta), int(512 * beta))
        self.Tconv3 = fuse_func(int(512 * beta), int(1024 * beta))
        self.Tconv4 = fuse_func(int(1024 * beta), int(2048 * beta))
        # for input in []:
        # self.slow_conv1_1 =  nn.Conv3d(3, self.fast_inplanes, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.pool2 = nn.MaxPool3d(kernel_size=(
                2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
    def _upsample(self, x, y):
        _, _, t, h, w = y.size()
        # print('spatial', x.shape, y.shape)
        x_upsampled = F.interpolate(x, [t, h, w], mode='nearest')

        return x_upsampled

    def _upsample_time(self, x):
        _,_,t, h, w = x.size()
        # print('time', x.shape)
        x_upsampled = F.interpolate(x, [t*2, h, w], mode='nearest')
        return x_upsampled

    def forward(self, input):
        fast, Tc = self.FastPath(input)
        # print('alpha',self.alpha)

        
        if self.slow_full_span:
            slow_input = torch.index_select(
                input,
                2,
                torch.linspace(
                    0,
                    input.shape[2] - 1,
                    input.shape[2] // self.alpha,
                ).long().cuda(),
            )
        else:
            slow_input = input[:, :, ::self.alpha, :, :]
        slow = self.SlowPath(slow_input, Tc)

        

        # print('1-before',slow[0].shape)
        # print('1-before',slow[1].shape)
        # print('1-before',slow[2].shape)
        # # fast[0] = self.pool2(fast[0])
        # fast[1] = self.pool2(fast[1])
        # fast[2] = self.pool2(fast[2])        

        slow[0] = self._upsample_time(slow[0])
        slow[1] = self._upsample_time(slow[1])
        slow[2] = self._upsample_time(slow[2])
        

        # print('1-fast',slow[0].shape)
        # print('1-fast',slow[1].shape)
        # print('1-fast',slow[2].shape)
        # print(rr)

        outFeat = []
        for sitem,fitem in zip(slow,fast):
            outFeat.append(torch.cat((sitem,fitem),1))
            # print(outFeat[-1].shape)
        return outFeat

    def SlowPath(self, input, Tc):
        # print('slowinpdi',input.shape)
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        # print('x',x.shape)
        x = torch.cat([x, Tc[0]], dim=1)
        x = self.slow_res1(x)
        x = torch.cat([x, Tc[1]], dim=1)
        c3 = self.slow_res2(x)
        x = torch.cat([c3, Tc[2]], dim=1)
        c4 = self.slow_res3(x)
        x = torch.cat([c4, Tc[3]], dim=1)
        c5 = self.slow_res4(x)

        return [c3,c4,c5]

    def FastPath(self, input):
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        x = self.fast_maxpool(x)
        Tc1 = self.Tconv1(x)
        x = self.fast_res1(x)
        Tc2 = self.Tconv2(x)
        c3 = self.fast_res2(x)
        Tc3 = self.Tconv3(c3)
        c4 = self.fast_res3(c3)
        Tc4 = self.Tconv4(c4)
        c5 = self.fast_res4(c4)
        return [c3,c4,c5], [Tc1, Tc2, Tc3, Tc4]

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1, dilation=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False
                )
            )

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, dilation=dilation, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, dilation=dilation, head_conv=head_conv))

        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1, dilation=1):
        downsample = None
        fused_inplanes = self.slow_inplanes + int(self.slow_inplanes * self.beta) * 2
        if stride != 1 or fused_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    fused_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False
                )
            )

        layers = []
        layers.append(block(fused_inplanes, planes, stride, downsample, dilation=dilation, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, dilation=dilation, head_conv=head_conv))

        return nn.Sequential(*layers)
```

The `SlowFast` class inherits from `nn.Module` and serves as a container for the SlowFast architecture. It takes several parameters in its constructor:

- `block`: The building block module for the network. It is a module that defines the internal structure of each residual block used in the network.
- `layers`: A list specifying the number of residual blocks in each stage of the network.
- `alpha`: The temporal downsampling rate used for the slow pathway. It determines the sampling rate at which frames are extracted from the input video.
- `beta`: The width factor that scales the number of channels in the network. It controls the capacity of the model.
- `fuse_only_conv`: A boolean indicating whether to use only convolutional layers for fusing information between the fast and slow pathways.
- `fuse_kernel_size`: The kernel size used in the fusion operation between the fast and slow pathways.
- `slow_full_span`: A boolean indicating whether to use the full temporal span of the slow pathway or subsample it.

The constructor initializes the different components of the SlowFast network, including the fast and slow pathways and the lateral connections. The fast pathway consists of a series of convolutional layers and residual blocks, while the slow pathway follows a similar structure but with additional temporal downsampling and fusion operations.

The `forward` method implements the forward pass of the SlowFast network. It takes an input video tensor and computes the feature representations in both the fast and slow pathways. The feature maps from the slow and fast pathways are concatenated and returned as the output.

The `SlowPath` and `FastPath` methods define the computation flow in the slow and fast pathways, respectively. They consist of a series of convolutional and residual blocks that process the input data and extract relevant features.

The `_make_layer_fast` and `_make_layer_slow` methods are helper functions used to create the sequence of residual blocks in the fast and slow pathways, respectively.

The `SlowFast` class implements the SlowFast architecture, which combines a fast pathway that processes frames at a high temporal resolution with a slow pathway that operates at a lower temporal resolution but with a larger context span. This architecture aims to capture both fast and slow motion information in videos for improved video understanding and action recognition.


(6) Part VI:
```python
def slowfast50(**kwargs):
    """Constructs a SlowFast-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
```

The `slowfast50` function takes keyword arguments `**kwargs` (which allows passing additional arguments) and returns a SlowFast-50 model. The function serves as a convenient way to create an instance of the `SlowFast` class with specific parameters.

Inside the function, the `SlowFast` class is instantiated with the following arguments:
- `Bottleneck`: The `Bottleneck` class, which represents the building block for each residual block in the SlowFast network. It is likely a specific implementation of the bottleneck architecture with skip connections.
- `[3, 4, 6, 3]`: A list specifying the number of residual blocks in each stage of the SlowFast network. In this case, it indicates that the SlowFast-50 model has 3 residual blocks in the first stage, 4 in the second stage, 6 in the third stage, and 3 in the fourth stage.
- `**kwargs`: This allows passing additional keyword arguments to the `SlowFast` class constructor. It enables flexibility in customizing the SlowFast-50 model with different parameters.

Finally, the constructed `model` is returned as the output of the `slowfast50` function.
The `slowfast50` function is a high-level wrapper that simplifies the creation of a SlowFast-50 model by instantiating the `SlowFast` class with specific parameters, including the choice of bottleneck block and the number of blocks in each stage.


### 3. About codes of `models/heads/acar.py` :

#### (1) Part I:
In this file, a class called `HR2O_NL` is defined, which represents a non-local module used in a neural network. 
```python
class HR2O_NL(nn.Module):
    def __init__(self, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(HR2O_NL, self).__init__()

        self.hidden_dim = hidden_dim

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        self.conv = nn.Conv2d(
            hidden_dim, hidden_dim,
            1 if mlp_1x1 else kernel_size,
            padding=0 if mlp_1x1 else padding,
            bias=False
        )
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)
        self.dp = nn.Dropout(0.2)
```

The `HR2O_NL` class is a subclass of `nn.Module` and is used to define a non-local module. It has the following components:

- `hidden_dim`: The dimensionality of the input feature maps.
- `conv_q`, `conv_k`, `conv_v`: Three convolutional layers used for computing the query, key, and value features. These layers take the input feature maps and transform them into the query, key, and value representations required for non-local operations.
- `conv`: A convolutional layer used for combining and transforming the non-local information. The kernel size and padding are determined based on the `mlp_1x1` parameter. If `mlp_1x1` is `True`, the kernel size is 1 and no padding is applied; otherwise, the kernel size and padding are set to the specified `kernel_size`.
- `norm`: A group normalization layer used for normalizing the virtual features after the non-local operation.
- `dp`: A dropout layer with a dropout rate of 0.2, applied to the virtual features.

```python
def forward(self, x):
    query = self.conv_q(x).unsqueeze(1)
    key = self.conv_k(x).unsqueeze(0)
    att = (query * key).sum(2) / (self.hidden_dim ** 0.5)
    att = nn.Softmax(dim=1)(att)
    value = self.conv_v(x)
    virt_feats = (att.unsqueeze(2) * value).sum(1)

    virt_feats = self.norm(virt_feats)
    virt_feats = nn.functional.relu(virt_feats)
    virt_feats = self.conv(virt_feats)
    virt_feats = self.dp(virt_feats)

    x = x + virt_feats
    return x
```

The `forward` method implements the forward pass of the `HR2O_NL` module. It takes an input tensor `x` and performs the following operations:

- The `x` tensor is passed through the `conv_q`, `conv_k`, and `conv_v` layers, producing the query, key, and value feature maps, respectively.
- The query and key feature maps are reshaped to match the dimensions required for the non-local operation. The query is unsqueezed along the second dimension, and the key is unsqueezed along the first dimension.
- The attention map is computed by element-wise multiplication of the query and key feature maps.

#### (2) Part II:
The code defines a class called `ACARHead`, which represents a head module used in a neural network. Here's a detailed breakdown of the code:

```python
class ACARHead(nn.Module):
    def __init__(self, width, roi_spatial=7, num_classes=60, dropout=0., bias=False,
                 reduce_dim=1024, hidden_dim=512, downsample='max2x2', depth=2,
                 kernel_size=3, mlp_1x1=False):
        super(ACARHead, self).__init__()

        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.MaxPool2d(roi_spatial)

        # actor-context feature encoder
        self.conv_reduce = nn.Conv2d(width, reduce_dim, 1, bias=False)

        self.conv1 = nn.Conv2d(reduce_dim * 2, hidden_dim, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, bias=False)

        # down-sampling before HR2O
        assert downsample in ['none', 'max2x2']
        if downsample == 'none':
            self.downsample = nn.Identity()
        elif downsample == 'max2x2':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # high-order relation reasoning operator (HR2O_NL)
        layers = []
        for _ in range(depth):
            layers.append(HR2O_NL(hidden_dim, kernel_size, mlp_1x1))
        self.hr2o = nn.Sequential(*layers)

        # classification
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(reduce_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None
```

The `ACARHead` class is a subclass of `nn.Module` and is used to define a head module in a neural network. It has the following components:

- `roi_spatial`: The spatial size of the region of interest (ROI).
- `roi_maxpool`: A max pooling layer used to pool the ROI features.
- `conv_reduce`: A convolutional layer used to reduce the dimensionality of the input features to `reduce_dim`.
- `conv1` and `conv2`: Two convolutional layers used to encode the actor-context features.
- `downsample`: A downsampling operation applied before the high-order relation reasoning operator. It can be either `'none'` or `'max2x2'`. If `'none'`, it is an identity operation. If `'max2x2'`, it performs max pooling with a kernel size of 3, stride of 2, and padding of 1.
- `hr2o`: A sequential module that contains multiple instances of the `HR2O_NL` module. The number of instances is determined by the `depth` parameter.
- `gap`: An adaptive average pooling layer used to pool the high-order features.
- `fc1` and `fc2`: Linear layers used for classification.
- `dp`: A dropout layer applied to the outputs if `dropout` is greater than 0, otherwise it is set to `None`.

```python
def forward(self, data):
    if not isinstance(data['features'], list):
        feats = [data['features']]
    else:
        feats = data['features']

    # temporal average pooling
    h, w = feats[0].shape[3:]
    # requires all features have the same spatial dimensions
    feats = [nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, f.shape[1], h, w) for f in feats]
    feats = torch.cat(feats, dim=1)

    feats = self.conv_reduce(feats)

    rois = data['rois']
    rois[:, 1] = rois[:, 1] * w
    rois[:, 2] = rois[:, 2] * h
    rois[:, 3] = rois[:, 3] * w
    rois[:, 4] = rois[:, 4] * h
    rois = rois.detach()
    roi_feats = torchvision.ops.roi_align(feats, rois, (self.roi_spatial, self.roi_spatial))
    roi_feats = self.roi_maxpool(roi_feats).view(data['num_rois'], -1)

    roi_ids = data['roi_ids']
    sizes_before_padding = data['sizes_before_padding']
    high_order_feats = []
    for idx in range(feats.shape[0]):  # iterate over mini-batch
        n_rois = roi_ids[idx+1] - roi_ids[idx]
        if n_rois == 0:
            continue

        eff_h, eff_w = math.ceil(h * sizes_before_padding[idx][1]), math.ceil(w * sizes_before_padding[idx][0])
        bg_feats = feats[idx][:, :eff_h, :eff_w]
        bg_feats = bg_feats.unsqueeze(0).repeat((n_rois, 1, 1, 1))
        actor_feats = roi_feats[roi_ids[idx]:roi_ids[idx+1]]
        tiled_actor_feats = actor_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats)
        interact_feats = torch.cat([bg_feats, tiled_actor_feats], dim=1)

        interact_feats = self.conv1(interact_feats)
        interact_feats = nn.functional.relu(interact_feats)
        interact_feats = self.conv2(interact_feats)
        interact_feats = nn.functional.relu(interact_feats)

        interact_feats = self.downsample(interact_feats)

        interact_feats = self.hr2o(interact_feats)
        interact_feats = self.gap(interact_feats)
        high_order_feats.append(interact_feats)

    high_order_feats = torch.cat(high_order_feats, dim=0).view(data['num_rois'], -1)

    outputs = self.fc1(roi_feats)
    outputs = nn.functional.relu(outputs)
    outputs = torch.cat([outputs, high_order_feats], dim=1)

    if self.dp is not None:
        outputs = self.dp(outputs)
    outputs = self.fc2(outputs)

    return {'outputs': outputs}
```

The `forward` method implements the forward pass of the `ACARHead` module. It takes a `data` dictionary as input, which contains the following keys: `'features'`, `'rois'`, `'num_rois'`, `'roi_ids'`, and `'sizes_before_padding'`. The method performs the following operations:

[1] If `data['features']` is a single tensor, it is wrapped in a list. Otherwise, it is assumed to be a list of tensors.
[2] Temporal average pooling is applied to each feature tensor in the list, ensuring that they all have the same spatial dimensions.
[3] The feature tensors are concatenated along the channel dimension and passed through the `conv_reduce` layer.
[4] The ROI coordinates in the `'rois'` tensor are adjusted to match the spatial dimensions of the feature tensors.
[5] The ROI features are obtained using the `roi_align` function from the `torchvision.ops` module. The features are then max pooled using `roi_maxpool` and reshaped.
[6] The iteration over the mini-batch begins. For each sample in the mini-batch:
   - The number of ROIs for the sample is determined.
   - Background features are extracted from the feature tensor and tiled to match the number of ROIs.
   - Actor features corresponding to the ROIs are selected.
   - The background and actor features are concatenated.
   - The concatenated features are passed through `conv1` and `conv2`, followed by ReLU activation.
   - The features are downsampled using the `downsample` operation.
   - The high-order relation reasoning operator (`hr2o`) is applied to the features.
   - The features are globally average pooled using `gap`.
   - The resulting features are appended to a list.
[7] The list of high-order features is concatenated along the batch dimension and reshaped.
[8] The ROI features are passed through `fc1` and ReLU activation.
[9] The ROI features and high-order features are concatenated along the channel dimension.
[10] If dropout is applied (`self.dp` is not None), the features are passed through the dropout layer.
[11] The final features are passed through `fc2` to obtain the classification outputs.
[12] The outputs are returned as a dictionary with the key `'outputs'`.

### 4. About codes of `models/heads/linear.py` :

The `LinearHead` class is a module that implements a linear classifier for object recognition. It takes input features and region of interest (ROI) information and produces classification outputs. Let's break down the code:
This module takes input features and ROI information, applies ROI pooling and linear classification, and produces class predictions. It can be used as a head module in an object recognition pipeline.

```python
class LinearHead(nn.Module):
    def __init__(self, width, roi_spatial=7, num_classes=60, dropout=0., bias=False):
        super(LinearHead, self).__init__()

        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.MaxPool2d(roi_spatial)

        self.fc = nn.Linear(width, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None
```

- The constructor initializes the `LinearHead` module.
- `width` represents the input feature dimension.
- `roi_spatial` defines the spatial size of the ROI.
- `num_classes` specifies the number of output classes.
- `dropout` controls the dropout probability (if applicable).
- `bias` determines whether to include bias in the linear layer.
- `self.roi_maxpool` is an instance of `nn.MaxPool2d` used for ROI max pooling.
- `self.fc` is a linear layer that maps input features to the number of classes.
- `self.dp` is a dropout layer that is applied to the ROI features.

```python
def forward(self, data):
    if not isinstance(data['features'], list):
        features = [data['features']]
    else:
        features = data['features']

    roi_features = []
    for f in features:
        sp = f.shape
        h, w = sp[3:]
        feats = nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, sp[1], h, w)

        rois = data['rois'].clone()
        rois[:, 1] = rois[:, 1] * w
        rois[:, 2] = rois[:, 2] * h
        rois[:, 3] = rois[:, 3] * w
        rois[:, 4] = rois[:, 4] * h
        rois = rois.detach()
        roi_feats = torchvision.ops.roi_align(feats, rois, (self.roi_spatial, self.roi_spatial))
        roi_feats = self.roi_maxpool(roi_feats).view(-1, sp[1])

        roi_features.append(roi_feats)

    roi_features = torch.cat(roi_features, dim=1)
    if self.dp is not None:
        roi_features = self.dp(roi_features)
    outputs = self.fc(roi_features)

    return {'outputs': outputs}
```

- The `forward` method implements the forward pass of the `LinearHead` module.
- It takes a `data` dictionary as input, which contains the keys `'features'` and `'rois'`.
- If `data['features']` is not a list, it is wrapped in a list. Otherwise, it is assumed to be a list of feature tensors.
- For each feature tensor, temporal average pooling is applied to reduce the temporal dimension to 1, resulting in 2D feature maps.
- The ROI coordinates in the `'rois'` tensor are adjusted to match the spatial dimensions of the feature maps.
- ROI align is performed using `torchvision.ops.roi_align` to extract ROI features from the feature maps.
- The ROI features are max pooled using `self.roi_maxpool` and reshaped.
- The ROI features from all feature tensors are concatenated along the channel dimension.
- If dropout is applied (`self.dp` is not None), the ROI features are passed through the dropout layer.
- The resulting features are passed through the linear layer `self.fc` to obtain the classification outputs.
- The outputs are returned as a dictionary with the key `'outputs'`.


### 5. About codes of `models/necks/bacis.py` :

The `BasicNeck` class is a module that performs pre-processing on input data for object detection. It applies augmentation, cropping, and jittering to generate region of interest (ROI) information for training or evaluation. 
This module takes input data, performs augmentation, cropping, and jittering, generates region of interest (ROI) information, and returns the processed data with additional metadata. It is typically used in an object detection pipeline.

```python
class BasicNeck(nn.Module):
    def __init__(self, aug_threshold=0., bbox_jitter=None, num_classes=60, multi_class=True):
        super(BasicNeck, self).__init__()

        self.aug_threshold = aug_threshold
        self.bbox_jitter = bbox_jitter
        self.num_classes = num_classes
        self.multi_class = multi_class
```

- The constructor initializes the `BasicNeck` module.
- `aug_threshold` is a threshold that determines the preserved ratio of bounding boxes after cropping augmentation.
- `bbox_jitter` is a configuration for bbox jittering (optional).
- `num_classes` specifies the number of classes.
- `multi_class` indicates whether the task is multi-class or not.

```python
def forward(self, data):
    rois, roi_ids, targets, sizes_before_padding, filenames, mid_times = [], [0], [], [], [], []
    bboxes, bbox_ids = [], []

    cur_bbox_id = -1
    for idx in range(len(data['aug_info'])):
        aug_info = data['aug_info'][idx]
        pad_ratio = aug_info['pad_ratio']
        sizes_before_padding.append([1. / pad_ratio[0], 1. / pad_ratio[1]])

        for label in data['labels'][idx]:
            cur_bbox_id += 1
            if self.training and self.bbox_jitter is not None:
                bbox_list = bbox_jitter(label['bounding_box'], self.bbox_jitter.get('num', 1), self.bbox_jitter.scale)
            else:
                bbox_list = [label['bounding_box']]

            for b in bbox_list:
                bbox = get_bbox_after_aug(aug_info, b, self.aug_threshold)
                if bbox is None:
                    continue
                rois.append([idx] + bbox)

                filenames.append(data['filenames'][idx])
                mid_times.append(data['mid_times'][idx])
                bboxes.append(label['bounding_box'])
                bbox_ids.append(cur_bbox_id)

                if self.multi_class:
                    ret = torch.zeros(self.num_classes)
                    ret.put_(torch.LongTensor(label['label']), torch.ones(len(label['label'])))
                else:
                    ret = torch.LongTensor(label['label'])
                targets.append(ret)

        roi_ids.append(len(rois))

    num_rois = len(rois)
    if num_rois == 0:
        return {'num_rois': 0, 'rois': None, 'roi_ids': roi_ids, 'targets': None, 
                'sizes_before_padding': sizes_before_padding,
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids}

    rois = torch.FloatTensor(rois).cuda()
    targets = torch.stack(targets, dim=0).cuda()
    return {'num_rois': num_rois, 'rois': rois, 'roi_ids': roi_ids, 'targets': targets, 
            'sizes_before_padding': sizes_before_padding,
            'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids}
```

- The `forward` method implements the forward pass of the `BasicNeck` module.
- It takes a `data` dictionary as input, which contains various keys like `'aug_info'`, `'labels'`, `'filenames'`, and `'mid_times'`.
- The method iterates over the `aug_info` list to process each augmentation information.
- For each `aug_info`, it retrieves the corresponding `pad_ratio` and appends it to the `sizes_before_padding` list.
- Then, for each label in `data['labels'][idx]`, it performs the following steps:
  - If the module is in training mode (`self.training`) and `bbox_jitter` is not None, it applies bbox jittering to the label's bounding box coordinates. Otherwise, it uses the original bounding box.
  - For each modified or original bounding box, it applies the cropping augmentation using `get_bbox_after_aug` function and the provided `aug_info` and `aug_threshold`. If the resulting bounding box is None (i.e., it is filtered out), it skips to the next bounding box.
  - If the bounding box is valid, it appends the ROI information, filenames, mid_times, bounding boxes, and bbox_ids to their respective lists.
  - For the `targets`, depending on whether it is a multi-class or single-class task, it creates a tensor with zeros for multi-class and a LongTensor for single-class, where the positions specified by `label['label']` are set to ones.
- After processing all labels, the method appends the current number of ROIs to `roi_ids`.
- Finally, it checks if any ROIs were generated (`num_rois`). If there are no ROIs, it returns a dictionary with appropriate keys and None values. Otherwise, it converts the lists to tensors, moves them to the GPU, and returns them as a dictionary.


### 6. About codes of `models/necks/utils.py` :

The `bbox_jitter` takes an input bounding box and applies jittering to generate one or multiple jittered bounding boxes. The jittering introduces small random perturbations to the original bounding box coordinates, allowing for data augmentation and increased robustness during training or evaluation tasks such as object detection.
The following code defines a function `bbox_jitter` that performs jittering on a bounding box. 

```python
def bbox_jitter(bbox, num, delta):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    if num == 1:
        jitter = np.random.uniform(-delta, delta, 4)
        bboxes = [[max(bbox[0] + jitter[0] * w, 0.), min(bbox[1] + jitter[1] * h, 1.),
                   max(bbox[2] + jitter[2] * w, 0.), min(bbox[3] + jitter[3] * h, 1.)]]
                   
        return bboxes
    
    bboxes = [bbox]
    jitter = np.random.uniform(-delta, delta, [num - 1, 4])
    for i in range(num - 1):
        bboxes.append([max(bbox[0] + jitter[i][0] * w, 0.), min(bbox[1] + jitter[i][1] * h, 1.),
                       max(bbox[2] + jitter[i][2] * w, 0.), min(bbox[3] + jitter[i][3] * h, 1.)])
    return bboxes
```

The function takes three parameters:
- `bbox`: The input bounding box coordinates represented as [x1, y1, x2, y2], where (x1, y1) denotes the top-left corner and (x2, y2) denotes the bottom-right corner.
- `num`: The number of jittered bounding boxes to generate. If `num` is 1, only the original bounding box with slight jitter is returned.
- `delta`: The maximum magnitude of jittering. The jittering values are sampled from a uniform distribution between `-delta` and `delta`.

Here's how the function works:

(1) It calculates the width `w` and height `h` of the input bounding box.
(2) If `num` is 1, it generates a single jittered bounding box:
   - It samples jittering values from a uniform distribution using `np.random.uniform(-delta, delta, 4)`.
   - The new bounding box coordinates are computed by adding the jittering values multiplied by the width or height to the original coordinates.
   - The resulting bounding box is clipped to ensure it falls within the range [0, 1] for each coordinate.
   - The single jittered bounding box is returned as a list of coordinates within a list.
(3) If `num` is greater than 1, it generates multiple jittered bounding boxes:
   - The original bounding box is appended to the list of bounding boxes.
   - It samples `num - 1` sets of jittering values using `np.random.uniform(-delta, delta, [num - 1, 4])`.
   - For each set of jittering values, it calculates new bounding box coordinates similar to the single jittered bounding box case.
   - The resulting bounding boxes are appended to the list.
(4) The function returns the list of generated bounding boxes.

Besides that function, there is also another function named `get_bbox_after_aug` that computes the adjusted bounding box coordinates after applying augmentation:

```python
def get_bbox_after_aug(aug_info, bbox, aug_threshold=0.3):
    if aug_info is None:
        return bbox
    
    cbox = aug_info['crop_box']
    w = cbox[2] - cbox[0]
    h = cbox[3] - cbox[1]
    
    l = max(min(bbox[0], cbox[2]), cbox[0])
    r = max(min(bbox[2], cbox[2]), cbox[0])
    t = max(min(bbox[1], cbox[3]), cbox[1])
    b = max(min(bbox[3], cbox[3]), cbox[1])
    
    if (b - t) * (r - l) <= (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) * aug_threshold:
        return None
    ret = [(l - cbox[0]) / w, (t - cbox[1]) / h, (r - cbox[0]) / w, (b - cbox[1]) / h]
    
    if aug_info['flip']:
        ret = [1. - ret[2], ret[1], 1. - ret[0], ret[3]]

    pad_ratio = aug_info['pad_ratio']
    ret = [ret[0] / pad_ratio[0], ret[1] / pad_ratio[1], ret[2] / pad_ratio[0], ret[3] / pad_ratio[1]]
    
    return ret
```

The function takes 3 parameters:
- `aug_info`: Information about the augmentation performed. It is expected to be a dictionary containing keys such as `'crop_box'`, `'flip'`, and `'pad_ratio'`.
- `bbox`: The original bounding box coordinates represented as [x1, y1, x2, y2], where (x1, y1) denotes the top-left corner and (x2, y2) denotes the bottom-right corner.
- `aug_threshold` (optional): A threshold value that determines whether the adjusted bounding box is significant enough after augmentation. Defaults to 0.3.

Here's how the function works:

1. It first checks if `aug_info` is `None`. If so, it returns the original bounding box as there is no augmentation applied.
2. It retrieves the crop box coordinates from `aug_info` as `cbox`, which represents the region where cropping occurred.
3. It calculates the width `w` and height `h` of the crop box.
4. It computes the adjusted coordinates of the bounding box by performing the following steps:
   - The left coordinate `l` is set to the maximum value between the minimum of `bbox[0]` and `cbox[2]` (right edge of crop box) and `cbox[0]` (left edge of crop box). This ensures that `l` falls within the crop box horizontally.
   - The right coordinate `r` is set to the maximum value between the minimum of `bbox[2]` and `cbox[2]` and `cbox[0]`. This ensures that `r` falls within the crop box horizontally.
   - The top coordinate `t` is set to the maximum value between the minimum of `bbox[1]` and `cbox[3]` (bottom edge of crop box) and `cbox[1]` (top edge of crop box). This ensures that `t`


### 7. About codes of `models/backbone_models.py` :
The following codes defines a function `backbone_models` that constructs a backbone model based on the provided arguments.

```python
def backbone_models(args):
    assert args.ARCH.startswith('resnet')

    modelperms = {'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3]}
    model_3d_layers = {'resnet50': [[0, 1, 2], [0, 2], [0, 2, 4], [0, 1]], 
                       'resnet101': [[0, 1, 2], [0, 2], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], [0, 1]]}
    assert args.ARCH in modelperms, 'Arch should be one of: ' + ','.join([m for m in modelperms])

    if args.MODEL_TYPE.endswith('-NL'):
        args.non_local_inds = [[], [1, 3], [1, 3, 5], []]
    else:
        args.non_local_inds = [[], [], [], []]

    base_arch, MODEL_TYPE = args.ARCH, args.MODEL_TYPE
    perms = modelperms[base_arch]

    args.model_perms = modelperms[base_arch]
    args.model_3d_layers = model_3d_layers[base_arch]

    model = resnetfpn(args)

    if args.MODE == 'train':
        if MODEL_TYPE.startswith('RCN'):
            model.identity_state_dict()
        if MODEL_TYPE.startswith('RCGRU') or MODEL_TYPE.startswith('RCLSTM'):
            model.recurrent_conv_zero_state()
        if not MODEL_TYPE.startswith('SlowFast'):
            load_dict = torch.load(args.MODEL_PATH)
            model.load_my_state_dict(load_dict)

    return model
```

The function takes an `args` object as input, which is expected to have several attributes including `ARCH`, `MODEL_TYPE`, and `MODE`. Here's how the function works:

(1) It checks if the `ARCH` attribute of `args` starts with `'resnet'`. If not, it raises an assertion error.
(2) It defines two dictionaries:
   - `modelperms` stores the number of blocks for each ResNet architecture. In this case, it contains configurations for `'resnet50'` and `'resnet101'`.
   - `model_3d_layers` stores the indices of 3D layers for each ResNet architecture. It specifies the layers that will be used for 3D feature extraction.
(3) It checks if the `ARCH` attribute is present in `modelperms`. If not, it raises an assertion error with a message listing the available architectures.
(4) Based on the `MODEL_TYPE` attribute, it determines the non-local indices for the backbone model. If `MODEL_TYPE` ends with `'-NL'`, it sets non-local indices accordingly; otherwise, it sets them to empty lists.
(5) It assigns values to `base_arch` and `MODEL_TYPE` based on the corresponding attributes in `args`.
(6) It retrieves the block permutations (`perms`) and 3D layer configurations (`model_3d_layers`) for the specified `base_arch`.
(7) It constructs a backbone model called `resnetfpn` with the provided arguments (`args`).
(8) If the mode (`MODE`) is set to `'train'`, it performs additional operations on the model based on the `MODEL_TYPE`:
   - If `MODEL_TYPE` starts with `'RCN'`, it calls the `identity_state_dict` method of the model, which initializes the model's state dictionary.
   - If `MODEL_TYPE` starts with `'RCGRU'` or `'RCLSTM'`, it calls the `recurrent_conv_zero_state` method of the model, which sets the initial states for recurrent convolutional layers to zero.
   - If `MODEL_TYPE` does not start with `'SlowFast'`, it loads the model's state dictionary from the specified `MODEL_PATH` using `torch.load` and assigns it to the model.
(9) Finally, it returns the constructed model.
