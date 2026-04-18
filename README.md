# PDF-Music-Sheet-reader
A PDF Music Sheet reader, with basic starts


The subfolder /lines has individual pictures of each line
The subfolder /linenotes has pictures of individual notes with the title containing information on how those should be put back together. The titles are in this format:
'media\\linenotes\\{self._linecountsubstring}_{notecount}_{clefofline}_{self._namesubstring}_v{self._countclefs}.png'

The subfolder /groundtruth has the textfile that is the key to differentiating garbage from not garbage
0: Garbage
1: NotGarbage

0: [1,0]
1: [0,1]