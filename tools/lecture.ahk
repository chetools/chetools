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



;specify window
^#!F10::
WinGet, id, ID, A
win_minimize := id		
Return


^#!F9::
If hidden
{
	WinShow, ahk_id %id%
	hidden := False
} Else {
	WinHide, ahk_id %id%
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

