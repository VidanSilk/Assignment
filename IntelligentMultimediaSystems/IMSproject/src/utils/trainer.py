import torch
from .progress import Bar, power_compress, power_uncompress, power_compress_mag, power_uncompress_mag
from .scores import cal_pesq_batch, cal_stoi_batch


######################################################################################################################
#                                               train loss function                                                  #
######################################################################################################################
def base_mag_train(model, train_loader, loss_calculator, optimizer, writer, EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()

    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        # generator
        input_mags, input_phase = model.stft(inputs)
        input_mags = power_compress_mag(input_mags, compression_factor=opt.cmpr)

        out_mags = model(input_mags)

        clean_mags, clean_phase = model.stft(targets)
        clean_mags = power_compress_mag(clean_mags)

        loss = loss_calculator(out_mags, clean_mags)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('mag', train_loss / batch_num, EPOCH)

    return train_loss


def base_complex_train(model, train_loader, loss_calculator, optimizer, writer,
                   EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()

    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        # generator
        input_specs = model.stft(inputs)
        input_real, input_imag = power_compress(input_specs, cut_len=opt.fft_len // 2 + 1, compression_factor=opt.cmpr)

        _, _, out_mags = model((input_real, input_imag))

        clean_specs = model.stft(targets)
        clean_real, clean_imag = power_compress(clean_specs, cut_len=opt.fft_len // 2 + 1, compression_factor=opt.cmpr)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

        loss = loss_calculator(out_mags, clean_mag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('mag', train_loss / batch_num, EPOCH)

    return train_loss


def joint_mag_train(model, train_loader, loss_calculator, optimizer, writer, EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    train_mag_loss = 0
    train_real_imag_loss = 0
    train_real_loss = 0
    train_imag_loss = 0
    batch_num = 0

    # train
    model.train()

    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        # generator
        input_mags, input_phase = model.stft(inputs)
        input_mags = power_compress_mag(input_mags, compression_factor=opt.cmpr)

        out_mags = model(input_mags)
        out_real = out_mags * torch.cos(input_phase)
        out_imag = out_mags * torch.sin(input_phase)

        clean_mags, clean_phase = model.stft(targets)
        clean_mags = power_compress_mag(clean_mags, compression_factor=opt.cmpr)
        clean_real = clean_mags * torch.cos(clean_phase)
        clean_imag = clean_mags * torch.sin(clean_phase)

        mag_loss = loss_calculator(out_mags, clean_mags)

        real_loss = loss_calculator(out_real, clean_real)
        imag_loss = loss_calculator(out_imag, clean_imag)
        real_imag_loss = real_loss + imag_loss

        loss = opt.c[0] * real_imag_loss + opt.c[1] * mag_loss #가중치

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mag_loss += mag_loss.item()
        train_real_imag_loss += real_imag_loss.item()
        train_real_loss += real_loss.item()
        train_imag_loss += imag_loss.item()
    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('mag+real+imag', train_loss, EPOCH)
    writer.log_train_loss('mag', train_mag_loss / batch_num, EPOCH)
    writer.log_train_loss('real+imag', train_real_imag_loss / batch_num, EPOCH)
    writer.log_train_loss('real', train_real_loss / batch_num, EPOCH)
    writer.log_train_loss('imag', train_imag_loss / batch_num, EPOCH)

    return train_loss


def joint_complex_train(model, train_loader, loss_calculator, optimizer, writer,
                             EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    train_mag_loss = 0
    train_real_imag_loss = 0
    train_real_loss = 0
    train_imag_loss = 0
    batch_num = 0

    # train
    model.train()

    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        # generator
        input_specs = model.stft(inputs)
        input_real, input_imag = power_compress(input_specs, cut_len=opt.fft_len // 2 + 1, compression_factor=opt.cmpr)

        out_real, out_imag, out_mags = model((input_real, input_imag))

        clean_specs = model.stft(targets)
        clean_real, clean_imag = power_compress(clean_specs, cut_len=opt.fft_len // 2 + 1, compression_factor=opt.cmpr)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

        mag_loss = loss_calculator(out_mags, clean_mag)

        real_loss = loss_calculator(out_real, clean_real)
        imag_loss = loss_calculator(out_imag, clean_imag)
        real_imag_loss = real_loss + imag_loss

        loss = opt.c[0] * real_imag_loss + opt.c[1] * mag_loss #가중치

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mag_loss += mag_loss.item()
        train_real_imag_loss += real_imag_loss.item()
        train_real_loss += real_loss.item()
        train_imag_loss += imag_loss.item()
    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('mag+real+imag', train_loss, EPOCH)
    writer.log_train_loss('mag', train_mag_loss / batch_num, EPOCH)
    writer.log_train_loss('real+imag', train_real_imag_loss / batch_num, EPOCH)
    writer.log_train_loss('real', train_real_loss / batch_num, EPOCH)
    writer.log_train_loss('imag', train_imag_loss / batch_num, EPOCH)

    return train_loss


######################################################################################################################
#                                               valid loss function                                                  #
######################################################################################################################
def base_mag_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()

    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            # generator
            input_mags, input_phase = model.stft(inputs)
            input_mags = power_compress_mag(input_mags, compression_factor=opt.cmpr)

            out_mags = model(input_mags)

            out_mags_uncompressed = power_uncompress_mag(out_mags, compression_factor=opt.cmpr)
            out_real = out_mags_uncompressed * torch.cos(input_phase)
            out_imag = out_mags_uncompressed * torch.sin(input_phase)
            out_specs = torch.cat([out_real, out_imag], dim=1)
            outputs = model.istft(out_specs)
            outputs = outputs.squeeze(1)

            clean_mags, clean_phase = model.stft(targets)
            clean_mags = power_compress_mag(clean_mags)

            loss = loss_calculator(out_mags, clean_mags)

            clean_wavs = targets.cpu().detach().numpy()[:, :outputs.size(1)]
            enhanced_wavs = outputs.cpu().detach().numpy()

            valid_loss += loss

            # get score
            pesq = cal_pesq_batch(clean_wavs, enhanced_wavs)
            stoi = cal_stoi_batch(clean_wavs, enhanced_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('mag', valid_loss / batch_num, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)
    writer.log_spectrogram(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi


def base_complex_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()

    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            # generator
            input_specs = model.stft(inputs)
            input_real, input_imag = power_compress(input_specs, cut_len=opt.fft_len // 2 + 1,
                                                    compression_factor=opt.cmpr)

            out_real, out_imag, out_mags = model((input_real, input_imag))
            out_specs = power_uncompress(out_real, out_imag, compression_factor=opt.cmpr)
            outputs = model.istft(out_specs)
            outputs = outputs.squeeze(1)

            clean_specs = model.stft(targets)
            clean_real, clean_imag = power_compress(clean_specs, compression_factor=opt.cmpr)
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

            loss = loss_calculator(out_mags, clean_mag)

            clean_wavs = targets.cpu().detach().numpy()[:, :outputs.size(1)]
            enhanced_wavs = outputs.cpu().detach().numpy()

            valid_loss += loss

            # get score
            pesq = cal_pesq_batch(clean_wavs, enhanced_wavs)
            stoi = cal_stoi_batch(clean_wavs, enhanced_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('mag', valid_loss / batch_num, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)
    writer.log_spectrogram(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi


def joint_mag_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    valid_mag_loss = 0
    valid_real_imag_loss = 0
    valid_real_loss = 0
    valid_imag_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()

    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            # generator
            input_mags, input_phase = model.stft(inputs)
            input_mags = power_compress_mag(input_mags, compression_factor=opt.cmpr)

            out_mags = model(input_mags)
            out_real = out_mags * torch.cos(input_phase)
            out_imag = out_mags * torch.sin(input_phase)

            clean_mags, clean_phase = model.stft(targets)
            clean_mags = power_compress_mag(clean_mags, compression_factor=opt.cmpr)
            clean_real = clean_mags * torch.cos(clean_phase)
            clean_imag = clean_mags * torch.sin(clean_phase)

            mag_loss = loss_calculator(out_mags, clean_mags)

            real_loss = loss_calculator(out_real, clean_real)
            imag_loss = loss_calculator(out_imag, clean_imag)
            real_imag_loss = real_loss + imag_loss

            loss = opt.c[0] * real_imag_loss + opt.c[1] * mag_loss #가중치

            out_specs = power_uncompress(out_real, out_imag)
            outputs = model.istft(out_specs)
            outputs = outputs.squeeze(1)
            clean_wavs = targets.cpu().detach().numpy()[:, :outputs.size(1)]
            enhanced_wavs = outputs.cpu().detach().numpy()

            valid_loss += loss
            valid_mag_loss += mag_loss
            valid_real_imag_loss += real_imag_loss
            valid_real_loss += real_loss
            valid_imag_loss += imag_loss

            # get score
            pesq = cal_pesq_batch(clean_wavs, enhanced_wavs)
            stoi = cal_stoi_batch(clean_wavs, enhanced_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('mag+real+imag', valid_loss, EPOCH)
    writer.log_valid_loss('mag', valid_mag_loss / batch_num, EPOCH)
    writer.log_valid_loss('real+imag', valid_real_imag_loss / batch_num, EPOCH)
    writer.log_valid_loss('real', valid_real_loss / batch_num, EPOCH)
    writer.log_valid_loss('imag', valid_imag_loss / batch_num, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)
    writer.log_spectrogram(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi


def joint_complex_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    valid_mag_loss = 0
    valid_real_imag_loss = 0
    valid_real_loss = 0
    valid_imag_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()

    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            # generator
            input_specs = model.stft(inputs)
            input_real, input_imag = power_compress(input_specs, cut_len=opt.fft_len // 2 + 1,
                                                    compression_factor=opt.cmpr)

            out_real, out_imag, out_mags = model((input_real, input_imag))
            out_specs = power_uncompress(out_real, out_imag, compression_factor=opt.cmpr)
            outputs = model.istft(out_specs)
            outputs = outputs.squeeze(1)

            clean_specs = model.stft(targets)
            clean_real, clean_imag = power_compress(clean_specs, cut_len=opt.fft_len // 2 + 1,
                                                    compression_factor=opt.cmpr)
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

            mag_loss = loss_calculator(out_mags, clean_mag)

            real_loss = loss_calculator(out_real, clean_real)
            imag_loss = loss_calculator(out_imag, clean_imag)
            real_imag_loss = real_loss + imag_loss

            loss = opt.c[0] * real_imag_loss + opt.c[1] * mag_loss #가중치

            clean_wavs = targets.cpu().detach().numpy()[:, :outputs.size(1)]
            enhanced_wavs = outputs.cpu().detach().numpy()

            valid_loss += loss
            valid_mag_loss += mag_loss
            valid_real_imag_loss += real_imag_loss
            valid_real_loss += real_loss
            valid_imag_loss += imag_loss

            # get score
            pesq = cal_pesq_batch(clean_wavs, enhanced_wavs)
            stoi = cal_stoi_batch(clean_wavs, enhanced_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('mag+real+imag', valid_loss, EPOCH)
    writer.log_valid_loss('mag', valid_mag_loss / batch_num, EPOCH)
    writer.log_valid_loss('real+imag', valid_real_imag_loss / batch_num, EPOCH)
    writer.log_valid_loss('real', valid_real_loss / batch_num, EPOCH)
    writer.log_valid_loss('imag', valid_imag_loss / batch_num, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)
    writer.log_spectrogram(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi

