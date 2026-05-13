#!/usr/bin/env python3
# convert_gauss_fix.py
# Modified generator: robust basis parsing + gen basis block output matching user's sample

import os, sys, copy

# --------------------------- user-adoptable settings ---------------------------
nbo = " pop=nbo" # not used directly in generator but kept
nmr = " nmr=giao"
polar = " polar prop=(potential,efg)"
solvent = " SCRF=(Solvent=acetonitrile,SMD)"
volume = " volume"

method = {
    "SP1": "# M06/Def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ pop=hirshfeld polar prop=efg IOp(3/174=1000000) IOp(3/175=2238200) IOp(3/177=452900) IOp(3/178=4655000) ",
    "SP2": "# M06/Def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ nmr=giao IOp(3/174=1000000) IOp(3/175=2238200) IOp(3/177=452900) IOp(3/178=4655000)  ",
    "SP4": "# M06/Def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ pop=NBO6 IOp(3/174=1000000) IOp(3/175=2238200) IOp(3/177=452900) IOp(3/178=4655000)  ",
    "POF": "# B3LYP gen 6D pseudo=read int=(grid=fine) empiricaldispersion=GD3BJ ",
}

sp_heavy_basis = "Def2TZVP"    # change for Ni sp jobs if needed

heavy_basis = """S    1   1.00
      7.6200000              1.0000000
S    1   1.00
      2.2940000              1.0000000
S    1   1.00
      0.8760000              1.0000000
S    1   1.00
      0.1153000              1.0000000
S    1   1.00
      0.0396000              1.0000000
P    1   1.00
     23.6600000              1.0000000
P    1   1.00
      2.8930000              1.0000000
P    1   1.00
      0.9435000              1.0000000
P    1   1.00
      0.0840000              1.0000000
P    1   1.00
      0.0240000              1.0000000
D    3   1.00
     42.7200000              0.0372699
     11.7600000              0.1956103
      3.8170000              0.4561273
D    1   1.00
      1.1690000              1.0000000
D    1   1.00
      0.2836000              1.0000000"""

ecp = """NI-ECP     2     10
d potential
  3
1    469.9324331            -10.0000000
2     85.4236411            -69.4084805
2     21.2674984            -12.0951020
s-d potential
  4
0    162.1686097              3.0000000
1    176.5333232             22.0253618
2     68.9562010            443.0181088
2     13.5792838            145.5696411
p-d potential
  4
0     69.0181506              5.0000000
1    275.5955596              4.9882824
2     47.1315453            256.6945853
2     12.9874075             78.4754450
"""

# Template for Link0 / resource lines (customize as you wish)
link0 = "%NProcShared=52\n$mem=16GB\n"
chkfolder = "%chk="

keywords = {
    0: "opt freq=noraman",
    1: "guess=read geom=check",
    2: "guess=read geom=check",
    3: "guess=read geom=check",
    4: "guess=read geom=check",
    5: "guess=read geom=check",
}

# Elements considered "heavy" for which we have explicit basis/ECP blocks
heavy = ["Ni"]
# default: elements that will use Def2TZVP for SP-only jobs (if gen not used)
heavy_sp = ["K"]

# --------------------------- helpers ---------------------------

def get_coms(dirpath):
    return [f for f in os.listdir(dirpath) if f.endswith((".gjf", ".com"))]

class Input:
    def __init__(self, filename, dirpath, route, jobs):
        self.file = os.path.join(dirpath, filename)
        self.name = filename[:-4].split("/")[-1]
        ok = self.get_coords()
        if not ok:
            self.name = False
            return
        # copy route dict (strings)
        self.sroute = copy.deepcopy(route)
        self.basis = {}
        # robust basis extraction
        for i in range(jobs):
            self.basis[i] = self._extract_basis_token(self.sroute[i])
        # detect heavy elements and prepare flags
        self.metaldetector()
    
    def _extract_basis_token(self, route_string):
        """
        Extract the basis token from a route string.
        If route contains 'METHOD/BASIS ...' returns BASIS.
        If route contains 'gen' returns 'gen'.
        Otherwise returns None or a safe string.
        """
        s = route_string.strip()
        # look for slash-style METHOD/BASIS
        if "/" in s:
            parts = s.split("/")
            if len(parts) >= 2:
                # take the token immediately after '/', first whitespace-delimited word
                token = parts[1].split()[0]
                return token
        # no slash-style basis: check for 'gen' or 'pseudo=read' etc.
        low = s.lower()
        if " gen " in low or low.startswith("gen ") or "pseudo=read" in low:
            return "gen"
        # fallback: attempt to find common basis names (simple heuristics)
        for candidate in ["Def2TZVP", "6-31g", "6-31g(d)", "Def2TZVPP"]:
            if candidate.lower() in s.lower():
                return candidate
        # final fallback:
        return None

    def get_coords(self):
        try:
            filecont = open(self.file, 'r').readlines()
        except Exception:
            return False
        # quick rejection for .mae etc
        if not filecont:
            return False
        if ".mae" in filecont[0]:
            return False
        self.coords = ""
        self.elements = []
        start = len(filecont)
        # find the geometry block heuristic: look for a line with exactly 2 tokens (charge,multiplicity hint) then coordinates after
        for l in range(len(filecont)-1):
            if len(filecont[l].split()) == 2 and len(filecont[l+1].split()) == 4:
                start = l+1
                # attempt to capture title (line before the two-token line)
                self.chsp = filecont[l].split()
                self.title = filecont[l-2].strip() if l-2 >= 0 else self.name
                break
        if start == len(filecont):
            return False
        end = None
        for l in range(start+1, len(filecont)):
            if len(filecont[l].split()) < 2:
                end = l
                break
        if end is None:
            # take until EOF
            end = len(filecont)
        for i in range(start, end):
            toks = filecont[i].split()
            # first token element, next three tokens coords
            if len(toks) >= 4:
                elem = toks[0].split("-")[0]
                self.elements.append(elem)
                self.coords += "{:<2}    {:>10}    {:>10}    {:>10}\n".format(elem, toks[1], toks[2], toks[3])
        return True

    def metaldetector(self):
        # create strings for heavy and other atoms
        self.heavyelems = " ".join(sorted(set([x for x in self.elements if x in heavy])))
        self.otherelems = " ".join(sorted(set([x for x in self.elements if x not in heavy])))
        self.heavy = 1 if len(self.heavyelems) > 0 else 0
        # debugging prints if desired (comment out in production)
        # print("HEAVY:", self.heavyelems, "OTHERS:", self.otherelems)
        return

def convert_gaussin(choice, dirpath):
    setup = {
        "P":   ["   phosphine opt+freq+4xSP\n\n", ["POF","SP1","SP2","SP4"]],
        "P2":  ["   phosphine opt+freq+4xSP\n\n", ["POF","SP4"]],
        "PdP": ["   Pd_phosphine opt+freq\n\n", ["POF"]],
        "PSP": ["   phosphine 3xSP\n\n", ["SP1","SP2","SP4"]],
        "PSP2":["   phosphine 2xSP\n\n", ["SP2","SP4"]],
        "Psolv":["  phosphine solvent SP\n\n",["SP1"]],
    }

    if choice not in setup:
        # fall back to prompting like original (but we won't prompt in batch usage)
        print("Choice must be one of:", ", ".join(setup.keys()))
        return

    methods_list = setup[choice][1]
    jobs = len(methods_list)
    # build route dict
    route = {}
    for i in range(jobs):
        mkey = methods_list[i]
        base_route = method[mkey]
        # append keyword block: for opt (i==0 usually) use keywords[0]; else guess=read geom=check
        kw = keywords[i] if i in keywords else ""
        # ensure spacing and newline termination
        route[i] = base_route + " " + kw + "\n\n"

    coms = get_coms(dirpath)
    if not coms:
        print("No .gjf or .com files found in", dirpath)
        return

    for com in coms:
        print("Processing:", com)
        data = Input(com, dirpath, route, jobs)
        if not data.name:
            print("Skipping (no coords found):", com)
            continue

        # Compose file content for this molecule: possibly multiple Link1 sections if multiple jobs
        filecontent = ""
        for i in range(jobs):
            # Link0 / resource lines + chk
            filecontent += link0
            filecontent += chkfolder + data.name + ".chk\n\n"
            # route line
            # ensure route starts with '#'
            r = data.sroute[i].strip()
            if not r.startswith("#"):
                r = "# " + r
            filecontent += r + "\n\n"
            # title
            filecontent += " " + data.title.strip() + "\n\n"
            # charge/mult line (we assume provided in input as two tokens earlier)
            if hasattr(data, "chsp"):
                filecontent += " ".join(data.chsp) + "\n"
            else:
                filecontent += "0 1\n"
            # coordinates: we only put coordinates for first job; subsequent jobs use guess=read geom=check
            if i == 0:
                filecontent += data.coords + "\n"
            else:
                filecontent += "\n"
            # If route uses gen (we detected earlier), append gen basis blocks
            if data.basis.get(i) == "gen":
                # collect element lists for gen basis section: other elements + heavy elements
                # The example writes element list header e.g. "N O H C Br 0"
                # Build unique element list preserving typical ordering: other elems then heavy
                elems_for_basis = sorted(set(data.elements))  # simple sorted; adjust ordering if desired
                # create element line as in sample where elements listed and then "0"
                filecontent += " ".join(elems_for_basis) + " 0\n"
                # here you probably want to select a light-atom basis; default to 6-31g(d) as in sample
                filecontent += "6-31g(d)\n"
                filecontent += "****\n"
                # If heavy atoms present, write the heavy custom basis (from heavy_basis string)
                if data.heavy:
                    # write heavy basis block header for Ni (example)
                    filecontent += data.heavyelems.upper() + " 0\n"
                    filecontent += heavy_basis + "\n"
                    filecontent += "****\n\n"
                    # then write ECP block for those atoms
                    filecontent += data.heavyelems.upper() + " 0\n"
                    filecontent += ecp + "\n****\n"
                else:
                    # no heavy; finish with separator
                    filecontent += "\n"
            else:
                # route does not use gen: write nothing extra (normal route with explicit basis used)
                # But if this is an optimization and heavy elements exist, include custom basis/ECP blocks
                if "opt" in r.lower() and data.heavy:
                    # write standard basis block for other elements, and heavy custom basis + ECP
                    if data.otherelems:
                        filecontent += data.otherelems + " 0\n" + sp_heavy_basis + "\n****\n"
                    filecontent += data.heavyelems + " 0\n" + heavy_basis + "\n****\n\n"
                    filecontent += data.heavyelems + " 0\n" + ecp + "\n****\n"
            # add Link1 separator only if there are more jobs to follow
            if i != jobs - 1:
                filecontent += "\n--Link1--\n\n"

        # write out .com (overwrite)
        outpath = os.path.join(dirpath, data.name + ".com")
        with open(outpath, 'w', newline='\n') as outf:
            outf.write(filecontent)
        print("Wrote:", outpath)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_gauss_fix.py <CHOICE>")
        print("Choices: P PSP PSP2 P2 PdP Psolv")
        sys.exit(1)

    choice = sys.argv[1]

    # *** THE FIX *** — always use current directory
    directory = os.getcwd()

    convert_gaussin(choice, directory)
