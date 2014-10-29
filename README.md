PBSC algorithm implementation
=============================

Version 0.92 (June 26, 2012)
Released under the BSD-license

More info: http://graal.ift.ulaval.ca/pbsc/

## Description
This program provides most of the source code used to obtain the empirical results published in the paper A PAC-Bayes Sample Compression Approach to Kernel Methods [1]. The algorithm details are in the Supplementary Materials [2].

This code has only been tested on Linux, but it should work on other platforms without too much effort. Feel free to contact me if you have questions, complaints or to urge me to publish a better version of this code! 

## Required Libraries
* GNU Scientific Library (GSL) -- Under Ubuntu, you can simply install the package "libgsl0-dev".

## How to make it works
* Download the source
* Run the "make" command in the source folder
* Go to the "bin" subfolder
* Execute the pbsc_align file to run PBSC-A learning algorithm.
    -- Basic example: ./pbsc_align -kernel.gamma 0.5 USvotes_train.dat
    -- Read usage instructions (pbsc_align-usage.txt) for more possibilities
* Execute the pbsc_nonalign file to run PBSC-N learning algorithm.
    -- Basic example: ./pbsc_nonalign -C 10 -kernel.gamma 0.5 USvotes_train.dat
    -- Read usage instructions (pbsc_nonalign-usage.txt) for more possibilities
* Execute the pbsc_classify file to classify a dataset with a learned classifier.
    -- Basic example: ./pbsc_classify USvotes_train.dat USvotes_test.dat
    -- Read usage instructions (pbsc_classify-usage.txt) for more possibilities

## Code Author
Pascal Germain, Groupe de Recherche en Apprentissage Automatique de l'Universite Laval (GRAAL) 
http://graal.ift.ulaval.ca/pgermain/
    
## References
[1] Pascal Germain, Alexandre Lacoste, François Laviolette, Mario Marchand, and Sara Shanian. A PAC-Bayes Sample Compression Approach to Kernel Methods. In ICML 2011. 
    http://www.icml-2011.org/papers/218_icmlpaper.pdf
    
[2] Pascal Germain, Alexandre Lacoste, François Laviolette, Mario Marchand, and Sara Shanian. A PAC-Bayes Sample Compression Approach to Kernel Methods: Supplementary Material. 2011. 
    http://graal.ift.ulaval.ca/public/papers/PBSC_suppmaterial.pdf