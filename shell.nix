let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (pykgs: [
	pykgs.opencv4 pykgs.numpy
    ]))
  ];
}

