workspaceselectionadd withinbonds 1 atom.ele Ni
workspaceselectionsubtract atom.ele N, Ni
delete atom.sel allowemptyentry=true
workspaceselectionreplace atom.ele Ni
fragment metal_centers
beginundoblock Add Fragment
fragmentadd metal_centers Square_Planar
endundoblock 
workspaceselectionreplace at.sel
beginundoblock Set Element
atom by=element element="Ni"
retypeset at.sel
endundoblock
workspaceselectionreplace atom.ele Ni
beginundoblock Increment Charge
formalcharge increment atom.selected
hydrogenapply atom.selected
endundoblock
beginundoblock Increment Charge
formalcharge increment atom.selected
hydrogenapply atom.selected
endundoblock
beginundoblock
workspaceselectionadd withinbonds 1 atom.ele Ni
workspaceselectionsubtract atom.ele N, Ni
endundoblock
workspaceselectionreplace at.sel
beginundoblock Set Element
atom by=element element="H"
retypeset at.sel
endundoblock
workspaceselectionreplace at.sel
beginundoblock Decrement Charge
formalcharge decrement atom.selected
hydrogenapply atom.selected
endundoblock
