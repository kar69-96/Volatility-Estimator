#!/usr/bin/env python3
"""Simple training script with cross-ticker training."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Subset

from src.models.chronos import ChronosVolatility
from src.training.data import prepare_raw_signal, compute_target, VolatilityDataset
from src.training.finetune import train
from src.models.base_model import get_device
from src.data.data_loader import get_market_data
from datetime import datetime, timedelta


def main():
    """Main training function."""
    # Full list of training tickers (S&P 500 companies and related)
    all_tickers_str = """A, AAPL, AAN, AAP, AAPL, AAXJ, ABBV, ABC, ABER, ABMD, ABT, ACGL, ACN,
ADBE, ADBE, ADM, ADP, ADSK, AEE, AFL, AGCO, AGI, AIG, AIZ, AJG, AKAM,
ALB, ALE, ALGN, ALK, ALL, ALLE, AMAT, AMCR, AMD, AME, AMGN, AMP, AMT,
AMZN, ANET, ANSS, AON, AOS, APA, APD, APH, APTV, ARE, ARNC, ASML, ATVI,
AVB, AVGO, AVY, AWK, AXP, AYI, AZO, BABA, BAC, BALL, BAM, BAX, BBWI,
BDX, BEN, BIIB, BK, BLK, BLL, BMC, BMY, BR, BRK.B, BSX, BSWN, C, CAG,
CAH, CARR, CAT, CB, CBOE, CBRE, CCI, CCK, CDNS, CE, CERN, CF, CFG, CHD,
CHRW, CHTR, CI, CINF, CL, CLX, CMCSA, CME, CMG, CMI, CMS, CNC, CNP, COF,
COST, COTY, CPB, CPRT, CRM, CSCO, CSX, CTAS, CTLT, CTRA, CTSH, CTVA, D,
DAL, DD, DHI, DHR, DIS, DISCA, DISCK, DLR, DLTR, DOV, DOW, DPZ, DRE,
DRI, DTE, DUOL, DVN, DXC, EA, EBAY, ECL, ED, EFT, EIX, EL, EMN, EMR, ENPH,
EOG, EQIX, EQR, ES, ESS, ETN, ETR, ETSY, EVRG, EW, EXC, EXPD, EXPE, EXR,
F, FANG, FAST, FBHS, FCX, FDX, FE, FFIV, FIS, FISV, FITB, FL, FLIR, FLS,
FMC, FOX, FOXA, FRC, FSLR, FTV, GD, GE, GILD, GIS, GL, GLW, GM, GOOG,
GOOGL, GPC, GPN, GPP, GPS, GRMN, GS, GWW, HAL, HAS, HBAN, HBI, HCA, HD,
HES, HFC, HIG, HII, HLT, HOLX, HON, HP, HPQ, HRL, HSY, HUM, IBM, ICE,
IFF, ILMN, INCY, INTC, INTU, IP, IPG, IPGP, IQV, IR, IRM, ISRG, IT, ITW,
IVZ, J, JBHT, JCI, JEF, JKHY, JNJ, JNPR, JPM, JWN, K, KEY, KEYS, KHC,
KIM, KLAC, KMB, KMI, KO, KR, KSS, L, LB, LEG, LHX, LIN, LKQ, LLY, LMT,
LNC, LNT, LRCX, LSTR, LULU, LVS, LW, LYB, M, MA, MAN, MAR, MAS, MCD,
MCHP, MCK, MCO, MDLZ, MDT, MET, MGM, MHK, MKC, MKTX, MLCO, MMC, MMM,
MNST, MO, MOS, MPC, MRK, MRO, MS, MSCI, MSFT, MSI, MTB, MTD, MU, MXIM,
MYL, NCLH, NEE, NEM, NFLX, NI, NKE, NLOK, NOC, NOV, NOW, NRG, NSC, NTAP,
NTRS, NUE, NVDA, NVR, NWL, NWS, NWSA, NXPI, O, ODFL, OGE, OHI, OLN, OMC,
ORCL, ORLY, OXY, PAYX, PBCT, PCAR, PCG, PEP, PFE, PFG, PG, PGR, PH, PKG,
PKI, PLD, PLTR, PMI, PNC, PNR, PPG, PPE, PPL, PRGO, PRU, PSA, PSX, PVH,
PXD, PYPL, QCOM, QRVO, RCL, RE, REG, REGN, RF, RHI, RJF, RL, RMD, ROK,
ROP, ROST, RSG, RTX, RUN, SBUX, SCHW, SEE, SEL, SJM, SLB, SNAP, SON, SPG,
SPGI, SRE, STE, STT, STZ, SWK, SWKS, SYK, SYY, T, TAP, TDG, TEL, TER,
TFC, TFSL, TGT, TJX, TMO, TMUS, TPR, TRMB, TROX, TRV, TSCO, TSLA, TT,
TVC, TYL, UAL, UDR, UHS, UIL, ULTA, UNH, UNM, UNP, UPS, URI, USB, V,
VFC, VLO, VMC, VRSK, VRSN, VRT, VTR, VTRS, VZ, WAB, WAT, WBD, WEC, WELL,
WFC, WHR, WM, WMB, WMT, WRB, WRK, WST, WY, XEL, XLNX, XOM, XRAY, XRX,
XYL, YUM, ZBRA, ZBH, ZION, ZTS"""
    
    # Parse ticker list and remove duplicates
    all_tickers = [t.strip() for t in all_tickers_str.replace('\n', ',').split(',')]
    all_tickers = [t for t in all_tickers if t]  # Remove empty strings
    all_tickers = list(dict.fromkeys(all_tickers))  # Remove duplicates while preserving order
    
    print(f"Total tickers in dataset: {len(all_tickers)}")
    
    # Split into train/val/test sets (80/10/10)
    np.random.seed(42)
    np.random.shuffle(all_tickers)
    
    train_size = int(0.8 * len(all_tickers))
    val_size = int(0.1 * len(all_tickers))
    
    train_tickers = all_tickers[:train_size]
    val_tickers = all_tickers[train_size:train_size + val_size]
    held_out_tickers = all_tickers[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Training tickers: {len(train_tickers)}")
    print(f"  Validation tickers: {len(val_tickers)}")
    print(f"  Held-out test tickers: {len(held_out_tickers)}")
    
    # Data directory
    data_dir = Path('data/cache')
    
    # Load and prepare training data (cross-ticker)
    train_datasets = []
    
    # Date range for downloading data (last 10 years should be sufficient)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    
    print(f"\nLoading/preparing training data for {len(train_tickers)} tickers...")
    for i, ticker in enumerate(train_tickers, 1):
        data_path = data_dir / f'{ticker}.parquet'
        
        # If cache file doesn't exist, download data
        if not data_path.exists():
            print(f"[{i}/{len(train_tickers)}] Downloading {ticker}...", end=' ')
            try:
                df = get_market_data(
                    symbol=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True,
                    cache_dir=str(data_dir),
                    cache_format='parquet'
                )
                print(f"✓ {len(df)} rows")
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        else:
            # Load from cache
            df = pd.read_parquet(data_path)
        
        # Ensure date column exists and set as index for alignment
        if 'date' in df.columns:
            # Convert date to datetime if needed, then set as index
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
            df = df.set_index('date').sort_index()
        elif df.index.name == 'date' or (hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index)):
            # Already has date index
            df = df.sort_index()
        
        # Compute returns
        returns = np.log(df['close'] / df['close'].shift(1))
        returns = returns.dropna()
        
        # Prepare raw signal (squared returns) - use the same index as returns
        raw_signal = returns ** 2
        
        # Compute target (log-realized variance)
        target = compute_target(returns, horizon=20)
        
        # Align raw_signal and target (both should have same index now)
        common_idx = raw_signal.index.intersection(target.index)
        raw_signal = raw_signal.loc[common_idx]
        target = target.loc[common_idx]
        
        # Create dataset
        dataset = VolatilityDataset(raw_signal, target, seq_length=60, horizon=20)
        if len(dataset) > 0:
            train_datasets.append(dataset)
            if i % 10 == 0 or i == len(train_tickers):
                print(f"  Processed {i}/{len(train_tickers)} tickers, {len(train_datasets)} successful, {sum(len(d) for d in train_datasets)} total samples")
    
    if not train_datasets:
        print("Error: No training datasets loaded!")
        return
    
    # Combine all training tickers
    combined_dataset = ConcatDataset(train_datasets)
    print(f"\n✓ Successfully loaded {len(train_datasets)}/{len(train_tickers)} training tickers")
    print(f"✓ Total training samples: {len(combined_dataset):,}")
    
    # Temporal split (time series aware)
    # For time series, we should split by time, not randomly
    # But for simplicity, we'll do a random split here
    # In production, use time-based split
    train_size = int(0.8 * len(combined_dataset))
    indices = torch.randperm(len(combined_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_data = Subset(combined_dataset, train_indices.tolist())
    val_data = Subset(combined_dataset, val_indices.tolist())
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Model
    device = get_device('auto')
    print(f"Using device: {device}")
    
    model = ChronosVolatility(use_lora=True).to(device)
    
    # Train
    print("\nStarting training...")
    model = train(model, train_loader, val_loader, epochs=50, lr=1e-4, device=device)
    
    # Save
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'chronos.pt'
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining complete. Model saved to {checkpoint_path}")
    
    # Test on held-out tickers
    print(f"\nPreparing {len(held_out_tickers)} held-out test tickers...")
    held_out_loaded = []
    for i, ticker in enumerate(held_out_tickers, 1):
        test_path = data_dir / f'{ticker}.parquet'
        if test_path.exists():
            try:
                test_df = pd.read_parquet(test_path)
                held_out_loaded.append(ticker)
            except Exception:
                pass
        else:
            print(f"[{i}/{len(held_out_tickers)}] Downloading held-out ticker {ticker}...", end=' ')
            try:
                test_df = get_market_data(
                    symbol=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True,
                    cache_dir=str(data_dir),
                    cache_format='parquet'
                )
                held_out_loaded.append(ticker)
                print(f"✓ {len(test_df)} rows")
            except Exception as e:
                print(f"✗ Error: {e}")
    
    print(f"Loaded {len(held_out_loaded)} held-out test tickers for evaluation")
    # TODO: Run evaluation on held-out tickers


if __name__ == '__main__':
    main()

