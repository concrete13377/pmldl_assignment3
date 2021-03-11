import random
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model import EncoderRNN, AttnDecoderRNN
from dataset import CustomDataset


def indexesFromSentence(lang, sentence):
    return [dataset.rus_w2i[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(dataset.end_of_string_token_idx)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def evaluate(encoder, decoder, sentence, max_length=max_len):
    max_length += 2
    with torch.no_grad():
        input_tensor = tensorFromSentence(dataset, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        max_length = min(max_len, input_length)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor(
            [[dataset.start_of_string_token_idx]], device=device
        )  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == dataset.end_of_string_token_idx:
                decoded_words.append("<end_of_string>")
                break
            else:
                decoded_words.append(dataset.eng_i2w[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[: di + 1]


def eval_test(sequence):
    for j, v in enumerate(sequence):
        try:
            dataset.rus_w2i[v]
        except:
            sequence[j] = "<pad>"

    l = len(sequence)
    if l != 9:
        val_ = (
            "<start_of_string> "
            + " ".join(sequence)
            + " "
            + ("<pad> " * (10 - l - 1))
            + "<end_of_string>"
        )
    else:
        val_ = "<start_of_string> " + " ".join(sequence) + " <end_of_string>"

    return " ".join(evaluate(encoder, decoder, val_)[0])


if __name__ == "__main__":
    global max_len
    max_len = 12
    dataset = CustomDataset("corpus.en_ru.1m.ru", "corpus.en_ru.1m.en", max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH = "Checkpoints"
    teacher_forcing_ratio = 0.5
    hidden_size = 256
    learning_rate = 0.1
    n_iters = 100

    encoder = EncoderRNN(dataset.rus_n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, dataset.eng_n_words, dropout_p=0.1).to(device)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=0)
    progress = tqdm(enumerate(dataset), total=len(dataset))
    EPOCHS = 3

    for epoch in range(EPOCHS):
        loss_total = 0
        for i, val in progress:
            input_tensor = torch.tensor(val["source"]).unsqueeze(1).to(device)
            target_tensor = torch.tensor(val["target"]).unsqueeze(1).to(device)

            encoder_hidden = encoder.initHidden()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_outputs = torch.zeros(
                max_len + 2, encoder.hidden_size, device=device
            )

            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden
                )
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor(
                [[dataset.start_of_string_token_idx]], device=device
            )

            decoder_hidden = encoder_hidden

            use_teacher_forcing = (
                True if random.random() < teacher_forcing_ratio else False
            )

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    loss += criterion(decoder_output, target_tensor[di])
                    decoder_input = target_tensor[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    topv, topi = decoder_output.topk(1)
                    decoder_input = (
                        topi.squeeze().detach()
                    )  # detach from history as input

                    loss += criterion(decoder_output, target_tensor[di])
                    if decoder_input.item() == dataset.end_of_string_token_idx:
                        break

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            if i % 1000 == 0:
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                    },
                    f"{PATH}/{i}.pt",
                )

            loss_total += loss.item() / target_length
        progress.set_description(loss_total)

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
        },
        f"{PATH}/final.pt",
    )

    with open("test.ru.txt") as f:
        test = f.readlines()

    with open("answer.txt", "w+") as f:
        for i, val in tqdm(enumerate(test), total=len(test)):
            val = dataset.norma_string(val).split()
            l = len(val)
            batches = l // 9 if l % 9 == 0 else l // 9 + 1
            seq = ""
            for b in range(batches):
                seq += eval_test(val[9 * b : 9 * (b + 1)]) + " "

            if i != len(test) - 1:
                f.write(seq + "\n")
            else:
                f.write(seq)
