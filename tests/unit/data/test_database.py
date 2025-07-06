"""
測試 database 模組
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime
from flask import Flask
from data.database import init_db, BaseModel, db


class TestDatabase:
    """Database 測試類"""
    
    @pytest.fixture
    def app(self):
        """創建測試應用"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app
    
    @patch('data.database.db')
    @patch('data.database.migrate')
    def test_init_db(self, mock_migrate, mock_db, app):
        """測試資料庫初始化"""
        # 調用 init_db
        result = init_db(app)
        
        # 驗證配置
        assert 'SQLALCHEMY_DATABASE_URI' in app.config
        assert app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] is False
        
        # 驗證初始化調用
        mock_db.init_app.assert_called_once_with(app)
        mock_migrate.init_app.assert_called_once_with(app, mock_db)
        
        # 驗證返回值
        assert result == mock_db
    
    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://test:test@localhost/testdb'})
    def test_init_db_with_env_database_url(self, app):
        """測試使用環境變數的資料庫 URL"""
        init_db(app)
        
        assert app.config['SQLALCHEMY_DATABASE_URI'] == 'postgresql://test:test@localhost/testdb'
    
    def test_init_db_default_sqlite(self, app):
        """測試預設 SQLite 資料庫"""
        init_db(app)
        
        assert 'sqlite:///' in app.config['SQLALCHEMY_DATABASE_URI']
        assert 'fire_guard.db' in app.config['SQLALCHEMY_DATABASE_URI']


class TestBaseModel:
    """BaseModel 測試類"""
    
    @pytest.fixture
    def mock_model(self):
        """創建模擬模型實例"""
        model = BaseModel()
        model.id = 1
        model.created_at = datetime(2024, 1, 1, 12, 0, 0)
        model.updated_at = datetime(2024, 1, 2, 12, 0, 0)
        
        # 模擬 __table__ 屬性
        mock_column1 = Mock()
        mock_column1.name = 'id'
        mock_column2 = Mock()
        mock_column2.name = 'created_at'
        mock_column3 = Mock()
        mock_column3.name = 'updated_at'
        
        model.__table__ = Mock()
        model.__table__.columns = [mock_column1, mock_column2, mock_column3]
        
        return model
    
    def test_to_dict(self, mock_model):
        """測試 to_dict 方法"""
        result = mock_model.to_dict()
        
        assert isinstance(result, dict)
        assert result['id'] == 1
        assert result['created_at'] == datetime(2024, 1, 1, 12, 0, 0)
        assert result['updated_at'] == datetime(2024, 1, 2, 12, 0, 0)
    
    @patch('data.database.db.session')
    def test_save(self, mock_session, mock_model):
        """測試 save 方法"""
        # 調用 save
        result = mock_model.save()
        
        # 驗證 session 調用
        mock_session.add.assert_called_once_with(mock_model)
        mock_session.commit.assert_called_once()
        
        # 驗證返回自身
        assert result == mock_model
    
    @patch('data.database.db.session')
    def test_delete(self, mock_session, mock_model):
        """測試 delete 方法"""
        # 調用 delete
        mock_model.delete()
        
        # 驗證 session 調用
        mock_session.delete.assert_called_once_with(mock_model)
        mock_session.commit.assert_called_once()
    
    def test_timestamps(self):
        """測試時間戳欄位"""
        # 驗證 created_at 和 updated_at 有預設值
        assert BaseModel.created_at.default is not None
        assert BaseModel.updated_at.default is not None
        assert BaseModel.updated_at.onupdate is not None
    
    @patch('data.database.db.session')
    def test_save_error_handling(self, mock_session, mock_model):
        """測試 save 錯誤處理"""
        # 模擬資料庫錯誤
        mock_session.commit.side_effect = Exception('Database error')
        
        # 驗證異常被拋出
        with pytest.raises(Exception) as exc_info:
            mock_model.save()
        
        assert 'Database error' in str(exc_info.value)
    
    @patch('data.database.db.session')
    def test_delete_error_handling(self, mock_session, mock_model):
        """測試 delete 錯誤處理"""
        # 模擬資料庫錯誤
        mock_session.commit.side_effect = Exception('Delete error')
        
        # 驗證異常被拋出
        with pytest.raises(Exception) as exc_info:
            mock_model.delete()
        
        assert 'Delete error' in str(exc_info.value)