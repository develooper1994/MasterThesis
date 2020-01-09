from Trainers.WaveGAN_Trainer import WaveGAN


def main():
    # TODO: Implement
    gan = WaveGAN()  # default behaviour is train.
    gan.train()


if __name__ == "__main__":
    print("Training Started")
    main()
    print("Training Ended")