from animal_classification.apis import inference_classifier, init_classifier

if __name__ == '__main__':
    checkpoint_path = 'work_dirs/mbv2_animal5_bs32/ckpt/network.pth'
    config_path = 'config/animal5/finetune/mbv2_animal5_bs32.py'
    img_path = [
        'data/Animal Classification/Interesting Images/Cat/Cat-Actress.jpeg',
        'data/Animal Classification/Interesting Images/Cow/Cow-bar.jpeg',
        'data/Animal Classification/Interesting Images/Dog/Dog-Art.jpeg',
        'data/Animal Classification/Interesting Images/Elephant/Elephant-Head.jpg',  # noqa: E501
        'data/Animal Classification/Interesting Images/Panda/Panda-Dress.jpeg',
    ]

    classifier = init_classifier(config=config_path,
                                 checkpoint=checkpoint_path)

    res = inference_classifier(classifier=classifier, imgs=img_path)
    print(res)
