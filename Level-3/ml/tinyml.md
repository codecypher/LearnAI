### TinyML

The key challenges in deploying neural networks on microcontrollers are the low memory footprint, limited power, and limited computation.

Perhaps the most obvious example of TinyML is within smartphones. These devices perpetually listen actively for ‘**wake words** such as “Hey Google” for Android smartphones, or ‘Hey Siri” on iPhones

Running these activities through the main central processing unit (CPU) of a smartphone (1.85 GHz for the modern iPhone) would deplete the battery in just a few hours which is not acceptable for something that most people would use a few times a day at most.

To combat this, developers created specialized low-power hardware that is able to be powered by a small battery (such as a circular CR2032 “coin” battery) to allow the circuits to remain active even when the CPU is not running which is basically whenever the screen is not lit.

These circuits can consume as little as 1 mW and can be powered for up to a year using a standard CR2032 batt

Thus, empowering edge devices with the capability of performing data-driven processing will produce a paradigm shift for industrial processes.

### How TinyML Works

TinyML algorithms work in much the same way as traditional machine learning models. Typically, the models are trained as usual on a user’s computer or in the cloud. Post-training is where the real tinyML work begins, in a process often referred to as deep compression.

### Model Distillation

After training, the model is altered in such a way as to create a model with a more compact representation. 

Pruning and knowledge distillation are two such techniques for this purpose.

