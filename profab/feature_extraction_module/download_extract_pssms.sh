#!/bin/sh


if [ -f all_pssm_matrices.tar.gz ]; then
	echo "Aldready downloaded !"
else
	echo "Downloading saved models"
        if [ -x "$(which wget)" ] ; then
	    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=13yXTmd4ErpuLmPIo7RO80zXK8rbYVSaG' -O all_pssm_matrices.tar.gz
	    echo "Download completed!"
	    echo "Extracting..."
            tar -xvf all_pssm_matrices.tar.gz
            echo "Extraction completed!."
            
	elif [ -x "$(which curl)" ] ; then
	    curl 'https://docs.google.com/uc?export=download&id=13yXTmd4ErpuLmPIo7RO80zXK8rbYVSaG' -O all_pssm_matrices.tar.gz
	    echo "Download completed!"
	    echo "Extracting..."
            tar -xvf all_pssm_matrices.tar.gz
            echo "Extraction completed!."
          
	else 
	    echo "***Please install wget or curl***"
	fi
        
fi







