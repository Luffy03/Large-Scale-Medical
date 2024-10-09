import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.criterion = nn.CrossEntropyLoss()
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()


def compute_loss(output, reports_ids, reports_masks, labels=None, vis_label=None, txt_label=None,
                 z_img=None, z_txt=None, args={}, similarity_function='dot'):
    criterion = LanguageModelCriterion()
    loss = criterion(output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()

    label_loss, match_loss = 0, 0
    if args.label_loss:
        label_criterion = torch.nn.BCEWithLogitsLoss()
        label_loss = label_criterion(vis_label, labels)
    if args.rank_loss:
        ranking_loss = RankingLoss()
        match_loss = ranking_loss(z_img, z_txt, labels, similarity_function)
    return loss + 0.1 * label_loss + 0.1 * match_loss


class RankingLoss(nn.Module):
    def __init__(self):
        super(RankingLoss, self).__init__()

    def forward(self, z_image, z_text, labels, similarity_function='dot'):

        return self.imposter_img_loss(z_image, z_text, labels, similarity_function) + \
               self.imposter_txt_loss(z_image, z_text, labels, similarity_function)

    def imposter_img_loss(self, z_image, z_text, labels, similarity_function):
        """
        A custom loss function for computing the hinge difference
        between the similarity of an image-text pair and
        the similarity of an imposter image-text pair
        where the image is an imposter image chosen from the batch
        """
        loss = torch.zeros(1, device=z_image.device, requires_grad=True)
        batch_size = z_image.size(0)

        for i in range(batch_size):
            # Select an imposter image index and
            # compute the maximum margin based on the image label difference
            j = i + 1 if i < batch_size - 1 else 0
            if torch.equal(labels[i], labels[j]):
                # This means the imposter image comes from the same acquisition
                margin = 0
            else:
                n = (labels[i].int() | labels[j].int()).sum().item()
                diff = (labels[i].int() ^ labels[j].int()).sum().item()
                margin = max(0.5, diff / n)

            if similarity_function == 'dot':
                paired_similarity = torch.dot(z_image[i], z_text[i])
                imposter_similarity = torch.dot(z_image[j], z_text[i])
            elif similarity_function == 'cosine':
                paired_similarity = \
                    torch.dot(z_image[i], z_text[i]) / (torch.norm(z_image[i]) * torch.norm(z_text[i]))
                imposter_similarity = \
                    torch.dot(z_image[j], z_text[i]) / (torch.norm(z_image[j]) * torch.norm(z_text[i]))
            elif similarity_function == 'l2':
                paired_similarity = -1 * torch.norm(z_image[i] - z_text[i])
                imposter_similarity = -1 * torch.norm(z_image[j] - z_text[i])

            diff_similarity = imposter_similarity - paired_similarity + margin
            if diff_similarity > 0:
                loss = loss + diff_similarity

        return loss / batch_size  # 'mean' reduction

    def imposter_txt_loss(self, z_image, z_text, labels, similarity_function):
        """
        A custom loss function for computing the hinge difference
        between the similarity of an image-text pair and
        the similarity of an imposter image-text pair
        where the text is an imposter text chosen from the batch
        """
        loss = torch.zeros(1, device=z_image.device, requires_grad=True)
        batch_size = z_image.size(0)

        for i in range(batch_size):
            # Select an imposter text index and
            # compute the maximum margin based on the image label difference
            j = i + 1 if i < batch_size - 1 else 0
            if torch.equal(labels[i], labels[j]):
                # This means the imposter image comes from the same acquisition
                margin = 0
            else:
                n = (labels[i].int() | labels[j].int()).sum().item()
                diff = (labels[i].int() ^ labels[j].int()).sum().item()
                margin = max(0.5, diff / n)

            if similarity_function == 'dot':
                paired_similarity = torch.dot(z_image[i], z_text[i])
                imposter_similarity = torch.dot(z_text[j], z_image[i])
            elif similarity_function == 'cosine':
                paired_similarity = \
                    torch.dot(z_image[i], z_text[i]) / (torch.norm(z_image[i]) * torch.norm(z_text[i]))
                imposter_similarity = \
                    torch.dot(z_text[j], z_image[i]) / (torch.norm(z_text[j]) * torch.norm(z_image[i]))
            elif similarity_function == 'l2':
                paired_similarity = -1 * torch.norm(z_image[i] - z_text[i])
                imposter_similarity = -1 * torch.norm(z_text[j] - z_image[i])

            diff_similarity = imposter_similarity - paired_similarity + margin
            if diff_similarity > 0:
                loss = loss + diff_similarity

        return loss / batch_size  # 'mean' reduction
