\begin{table}[]
\centering
\caption{My caption}
\label{my-label}
\resizebox{\textwidth}{!}{ \begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c|}
\cline{2-13}
                                    & \multicolumn{4}{c|}{Wdbc}             & \multicolumn{4}{c|}{Movement\_Libras} & \multicolumn{4}{c|}{Arrhythmia}       \\ \cline{2-13}
                                    & \% clas in & \% clas out & \% red & T & \% clas in & \% clas out & \% red & T & \% clas in & \% clas out & \% red & T \\ \hline
{{#items}}
\multicolumn{1}{|c|}{ {{name}} } &  {{W_clas_in}}    & {{W_clas_out}}  & {{W_red}}  & {{W_T}} & {{M_clas_in}} & {{M_clas_out}}  & {{M_red}}  & {{M_T}} & {{A_clas_in}} & {{A_clas_out}} & {{A_red}} & {{A_T}}  \\ \hline
{{/items}}
\end{tabular} }
\end{table}
