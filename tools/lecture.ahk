#SingleInstance Force

global win_minimize

^#!r::
Send, ^s ; To save a changed script
Sleep, 300 ; give it time to save the script
Reload
Return

^!#e::Edit

+WheelDown::WheelRight
+WheelUp::WheelLeft



;specify windows
^#!F10::
WinGet, id1, ID, A		
Return


^#!F9::
If hidden
{
	WinShow, ahk_id %id1%
	hidden := False
} Else {
	WinHide, ahk_id %id1%
	hidden := True
}
Return

;specify windows
^#!F8::
WinGet, id2, ID, A		
Return


^#!F7::
If hidden
{
	WinShow, ahk_id %id2%
	hidden := False
} Else {
	WinHide, ahk_id %id2%
	hidden := True
}
Return


::nplot::
(
import numpy as np
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default='plotly_dark'
)

