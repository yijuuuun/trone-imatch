"""Conditional follow-up: severe Gamma/CLAHE did not restore daylight Pilot success."""
import pilot_matrix as matrix

matrix.CONDITIONS.append(dict(id='E_severe_gamma_clahe',suite='E',precision='fp16',
    headings=8,scale=1.,light='severe',enhance='gamma_clahe'))

if __name__=='__main__':matrix.main()
