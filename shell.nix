# shell.nix
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
      # select Python packages here
      numpy
      ipykernel
      matplotlib
      pip
      jupyter
      notebook
      ultralytics
      pyyaml
    ]))
  ];
}
