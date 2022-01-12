#!/bin/sh

if [ -f profab/feature_extraction_module/swissprot_pssms.zip ]; then
	echo "Aldready downloaded!"
else
	echo "Downloading..."
	if [ -x "$(which wget)" ] ; then
	    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1AeDvRgxMIViUz0hKz08UazisseTkvcv9' -O profab/feature_extraction_module/swissprot_pssms.zip
	    echo "Download completed!"
	    
	elif [ -x "$(which curl)" ] ; then
	    curl 'https://docs.google.com/uc?export=download&id=1AeDvRgxMIViUz0hKz08UazisseTkvcv9' -O profab/feature_extraction_module/swissprot_pssms.zip
	    echo "Download completed!"
	else 
	    echo "***Please install wget or curl***"
	fi
        
fi







