green=`tput setaf 2`
cyan=`tput setaf 6`
reset=`tput sgr0`
find tests -name '*tests.py' -print0 |
    while IFS= read -r -d '' line; do
        echo "${line}"
        if [[ $line = "tests/filter_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/filtering/filter_options.py
        elif [[ $line = "tests/signal_loading_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/signal_loading/channel_options.py
            python3 -m coverage report -m epviz/signal_loading/channel_info.py
        elif [[ $line = "tests/plot_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/plot.py
        elif [[ $line = "tests/plot_utils_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/plot_utils.py
        elif [[ $line = "tests/stats_fs_band_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/signal_stats/signalStats_options.py
            python3 -m coverage report -m epviz/signal_stats/signalStats_info.py
        elif [[ $line = "tests/edf_saving_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/edf_saving/saveEdf_options.py
            python3 -m coverage report -m epviz/edf_saving/saveEdf_info.py
            python3 -m coverage report -m epviz/edf_saving/anonymizer.py
        elif [[ $line = "tests/spectrogram_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/spectrogram_window/spec_options.py
        elif [[ $line = "tests/prediction_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/predictions/prediction_options.py
            python3 -m coverage report -m epviz/predictions/prediction_info.py
        elif [[ $line = "tests/edf_loading_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/preprocessing/edf_loader.py
        elif [[ $line = "tests/image_saving_tests.py" ]]
        then
            echo "${green} Running tests for ${line} ${reset}"
            python3 -m coverage run "$line"
            echo "${cyan} Coverage Report for ${line}"
            python3 -m coverage report -m epviz/image_saving/saveImg_options.py
            python3 -m coverage report -m epviz/image_saving/saveTopoplot_options.py
        fi
    done
echo "${reset}"
