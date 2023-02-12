#!/usr/bin/env python
docstring='''string Template for HTML output'''
from string import Template

index_template=Template('''<html>
<head><meta http-equiv="content-type" content="text/html; charset=UTF-8">
<title>ATGO result for $TITLE_NAME</title></head>
<body>
<script type="text/javascript" src="$SRC_DIR/$JAVASCRIPT0"></script>
<script type="text/javascript" src="$SRC_DIR/$JAVASCRIPT1"></script>
[<a href="$HTTP_LINK">back to server</a>]<br/>
<h1 align="center">ATGO result for protein $JOBID</h1>
<h4 align="center">[Download <a href="result.zip" download>result.zip</a>
for all prediction results] </h4>
<font size="4" face="Arial">

<div style="background:#6599FF;width:500px;"><b><font face="Arial" 
color="#FFFFFF" size=4>&nbsp;User Input</font></b></div>
<ul>
    <table style="font-family:Monospace;font-size:14px;background:#F2F2F2;">
    <tr><td colspan="2">&gt;$TARGET_NAME ($LEN residues)<br>
    $WRAP_SEQUENCE<br>
    </td></tr>
    </table><br>
    <font size=2>
    Download query <a href="$FASTA_INPUT" download>sequence</a> 
    </font>
</ul>

<div style="background:#6599FF;width:500px;"><b><font face="Arial" 
color="#FFFFFF" size=4>&nbsp;Predicted Gene Ontology (GO) Terms</font></b></div>
<ul>
<table border="0" style="background:#F2F2F2";align="left"><tr><td>
$MF_TABLE
$BP_TABLE
$CC_TABLE
</td></tr>
</table>
</ul>

</font>
<h4 align="center">[Download <a href="result.zip" download>result.zip</a>
for all prediction results]</h4>
<hr>
<b>Reference:</b>
<ul><font size=3><font face = Monospace>$CITATION</ul>
[<a href="$HTTP_LINK">back to server</a>]<br/>
</body></html>
''')# JAVASCRIPT0,JAVASCRIPT1 - JSmol script
    # JAVASCRIPT2   - 3Dmol script
    # JOBID         - job ID
    # CITATION      - MetaGO paper
    # HTTP_LINK     - webserver link
    # TARGET_NAME   - target name
    # LEN           - sequence length
    # WRAP_SEQUENCE - wrapped sequence
    # FASTA_INPUT   - input fasta sequence file
    # PDB_INPUT     - input PDB structure file
    # TEMPLATE_PDB_FILE  - first template PDB file
    # JMOL_ROOT     - root dir of JSmol
    # SIZE          - JSmol viewer size
    # TEMPLATE_BUTTONS   - list of top structure analogs
    # EC_TABLE           - table showing EC prediction
    # MF_TABLE, BP_TABLE, CC_TABLE - table showing GO prediction
    # LBS_TABLE          - table showing LBS prediction


go_table_template=Template('''
<table border="0" style="background:#F2F2F2";align="left"><tr>
<td align ="center" style="background:#FFFFFF">
    <a href="$DAG_GRAPH" target="_blank">
    <img width=$SIZE height=$SIZE border="1" style="border-color:black" alt="$DAG_GRAPH" src="$SRC_DIR/$DAG_GRAPH">
    </a>
</td>
<td valign="top" align="left">
    <table border="0">
        <tr><td><b>$NAMESPACE ($ASPECT)</b><br></tr></td>
    </table>
    <table border="0" style="font-family:Arial;font-size:13px;">
        <tr><td align="left"><b>GO term</b></td><td align="center"><b>Cscore<sup>GO</sup></b></td><td><b>Name</b></td></tr>
        $GO_BUTTONS
    </table><br>
    <table border="0" style="font-family:Arial;font-size:13px;">
        <tr><td>Download <a href="$PRED_FILE" download>full result</a> of the above consensus prediction.</tr></td>
    </table><br>
    <table cellspacing="2" border="0" style="font-family:Arial;font-size:12px;">
        <tr><td valign="top"></td><td align="justify"><b>Click the graph to show a high resolution version.</b></td></tr>
        <tr><td valign="top">(a)</td><td align="justify">Cscore<sup>GO</sup> is the confidence score of predicted GO terms. Cscore<sup>GO</sup> values range in between [0-1]; where a higher value indicates a better confidence in predicting the function using the template.</td></tr>
        <tr><td valign="top">(b)</td><td align="justify">The graph shows the predicted terms within the Gene Ontology hierachy for $NAMESPACE. Confidently predicted terms are color coded by Cscore<sup>GO</sup>:</td></tr>
        <tr><td valign="top"></td><td align="justify"><table cellspacing="2" border="0" style="border-color:black;font-family:Arial;font-size:12px;"><tr>$COLORBAR</tr></table></td></tr>
     </table>
</td>
</tr>
</table>
''') # NAMESPACE  - full name of GO aspect
     # ASPECT     - short name of GO aspect
     # DAG_GRAPH  - image file for directed acyclic graph (DAG)
     # SIZE       - image size
     # GO_BUTTONS - list of predicted GO terms
     # COLORBAR   - table showing DAG color code

go_button_template=Template('''
        <tr><td align="left"><a href="$GO_LINK$TERM" target="_blank">$TERM</a></td><td align="center">$CSCORE</td><td>$NAME</td></tr>'''
)   # GO_LINK - http link to AmiGO website for each GO term entry
    # TERM    - GO term
    # CSCORE  - confidence score
    # NAME    - defination of GO term


